"""FastAPI dependency injection for platform services."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml
from sqlalchemy.orm import Session, sessionmaker

from platform.config.loader import ConfigLoader
from platform.extensions.catalog import ExtensionCatalog
from platform.extensions.installer import PackageInstaller
from platform.extensions.manager import CapabilityManager
from platform.persistence.repositories.package_repo import InstalledExtensionStore
from platform.core.interfaces.llm import ILLMProvider
from platform.core.models.tool import AdapterType, ToolDefinition
from platform.hitl.approval_manager import ApprovalManager
from platform.knowledge.config import KnowledgeConfig
from platform.knowledge.embedder import OpenAIEmbedder
from platform.knowledge.indexer import KnowledgeIndexer
from platform.knowledge.retriever import KnowledgeRetriever
from platform.knowledge.service import KnowledgeService
from platform.knowledge.vector_store import FAISSVectorStore
from platform.llm.openai_provider import OpenAIProvider
from platform.memory.in_memory_store import InMemoryStore
from platform.observability.composite_observer import CompositeObserver
from platform.observability.console_observer import ConsoleObserver
from platform.observability.persisting_observer import PersistingObserver
from platform.orchestrator.orchestrator import Orchestrator
from platform.orchestrator.run_manager import RunManager
from platform.persistence.database import Base, build_engine, build_session_factory
from platform.persistence.repositories.plan_repo import PlanRepository
from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.execution_adapter import ExecutionAdapter
from platform.planner.planner_service import PlannerService
from platform.tools.filesystem_adapter import FilesystemAdapter
from platform.policy.engine import PolicyEngine
from platform.registries.agent_registry import AgentRegistry
from platform.registries.tool_registry import ToolRegistry
from platform.registries.workflow_registry import WorkflowRegistry
from platform.state.shared_state import SharedState

_extension_catalog: ExtensionCatalog | None = None
_installed_extension_store: InstalledExtensionStore | None = None
_package_installer: PackageInstaller | None = None
_orchestrator: Orchestrator | None = None
_run_manager: RunManager | None = None
_hitl_manager: ApprovalManager | None = None
_workflow_registry: WorkflowRegistry | None = None
_tool_registry: ToolRegistry | None = None
_session_factory: sessionmaker[Session] | None = None
_knowledge_service: KnowledgeService | None = None
_knowledge_indexer: KnowledgeIndexer | None = None
_capability_registry: CapabilityRegistry | None = None
_capability_manager: CapabilityManager | None = None
_planner_service: PlannerService | None = None
_execution_adapter: ExecutionAdapter | None = None


def initialize(
    workflows_dir: Path,
    knowledge_config_path: Path | None = None,
) -> None:
    """Create and wire all platform singletons. Called once from lifespan."""
    global _extension_catalog, _installed_extension_store, _package_installer
    global _orchestrator, _run_manager, _hitl_manager, _workflow_registry
    global _tool_registry, _session_factory, _knowledge_service, _knowledge_indexer
    global _capability_registry, _capability_manager, _planner_service, _execution_adapter

    # 1a. Extension catalog — loaded from static YAML manifests; no DB required
    _extension_catalog = ExtensionCatalog.load(Path("resources/extensions"))

    # 1. Database — must come first so knowledge stack can use session_factory
    database_url = os.environ.get("DATABASE_URL", "sqlite:///./workflow.db")
    engine = build_engine(database_url)
    Base.metadata.create_all(engine)
    session_factory = build_session_factory(engine)
    _session_factory = session_factory

    # Phase C startup migration: upgrade any preview_only plans to pending_review.
    # preview_only was created before generated agents were executable; it no longer exists.
    _migration_repo = PlanRepository()
    with session_factory() as _mig_session:
        _count = _migration_repo.upgrade_preview_only_to_pending_review(_mig_session)
        _mig_session.commit()
    if _count:
        print(f"[Planner] Migrated {_count} preview_only plan(s) to pending_review.", file=sys.stderr)

    # 2. Knowledge stack — built before ConfigLoader so it can be wired to tools
    cfg_path = knowledge_config_path if knowledge_config_path is not None else Path("knowledge_config.yaml")
    service, indexer = _build_knowledge_stack(cfg_path, session_factory)
    _knowledge_service = service
    _knowledge_indexer = indexer

    # 3. Workflow registries
    wf_registry = WorkflowRegistry()
    ag_registry = AgentRegistry()
    tl_registry = ToolRegistry()
    ConfigLoader(
        wf_registry, ag_registry, tl_registry, knowledge_service=service
    ).load_all(workflows_dir)

    # Register filesystem tool in the runtime registry so generated agents can call it.
    tl_registry.register(
        "filesystem_read_file",
        FilesystemAdapter(),
        ToolDefinition(
            name="filesystem_read_file",
            description="Reads a local file and returns its UTF-8 contents.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read, relative to the working directory",
                    }
                },
                "required": ["path"],
            },
            adapter_type=AdapterType.FILESYSTEM,
        ),
    )

    memory_store = InMemoryStore()
    shared_state = SharedState()
    policy_engine = PolicyEngine()
    observer = CompositeObserver([
        ConsoleObserver(),
        PersistingObserver(session_factory),
    ])
    run_manager = RunManager()
    hitl_manager = ApprovalManager(run_manager)

    llm_provider = _create_llm_provider()

    orchestrator = Orchestrator(
        workflow_registry=wf_registry,
        agent_registry=ag_registry,
        tool_registry=tl_registry,
        memory_store=memory_store,
        policy_engine=policy_engine,
        observer=observer,
        run_manager=run_manager,
        llm_provider=llm_provider,
        shared_state=shared_state,
    )

    _orchestrator = orchestrator
    _run_manager = run_manager
    _hitl_manager = hitl_manager
    _workflow_registry = wf_registry
    _tool_registry = tl_registry

    # V3 planner stack
    cap_registry = CapabilityRegistry.build_pr_review_registry()
    # NOTE: filesystem_read capability is intentionally NOT pre-registered here.
    # The marketplace starts empty; capabilities are added only when the user installs
    # the filesystem-reader extension via POST /extensions/install. This allows
    # CapabilityManager.resolve() to correctly detect missing capabilities and surface
    # install suggestions for goals requiring filesystem_read.
    _capability_registry = cap_registry
    _capability_manager = CapabilityManager(
        catalog=_extension_catalog,
        capability_registry=cap_registry,
    )
    _planner_service = PlannerService(
        llm=llm_provider,
        registry=cap_registry,
        capability_manager=_capability_manager,
    )
    _execution_adapter = ExecutionAdapter(
        orchestrator=orchestrator,
        workflow_registry=wf_registry,
        agent_registry=ag_registry,
        capability_registry=cap_registry,
    )

    # Package system — wired after V3 registries so restore_from_db() adds
    # marketplace tools idempotently on top of the V3 baseline.
    _installed_extension_store = InstalledExtensionStore()
    _package_installer = PackageInstaller(
        catalog=_extension_catalog,
        store=_installed_extension_store,
        tool_registry=tl_registry,
        capability_registry=cap_registry,
    )
    with session_factory() as _startup_session:
        # Auto-install platform default extensions on first startup.
        # Subsequent startups skip auto-install because is_installed() returns True.
        _auto_install_defaults(_startup_session)
        _startup_session.commit()
        # Restore all active packages (including auto-installed ones). Static-agent
        # packages are no-ops in restore; runtime_agent packages re-register tools.
        _package_installer.restore_from_db(_startup_session)


def _build_knowledge_stack(
    config_path: Path,
    session_factory: sessionmaker[Session],
) -> tuple[KnowledgeService | None, KnowledgeIndexer | None]:
    """Build the full knowledge stack from config. Returns (None, None) if unavailable."""
    if not config_path.exists():
        print(
            f"[WARNING] {config_path} not found; knowledge features disabled.",
            file=sys.stderr,
        )
        return None, None
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = KnowledgeConfig.from_dict(data)
        embedder = OpenAIEmbedder(model=config.embedding_model)
        vs = FAISSVectorStore(Path(config.vector_store_path))
        retriever = KnowledgeRetriever(embedder, vs)
        service = KnowledgeService(retriever)
        indexer = KnowledgeIndexer(config, embedder, vs, session_factory)
        return service, indexer
    except Exception as exc:
        print(
            f"[WARNING] Knowledge initialization failed: {exc}",
            file=sys.stderr,
        )
        return None, None


def _auto_install_defaults(session: Session) -> None:
    """Install platform default extensions on first startup.

    GitHub Integration and Knowledge Search are auto-installed so they appear in the
    marketplace as installed from day one. They use the static_agent path, so no runtime
    registration occurs here — the V3 path (ConfigLoader + build_pr_review_registry)
    continues to own that. Subsequent startups skip auto-install because is_installed()
    returns True.

    Knowledge Search is conditional: if knowledge_config.yaml is absent or initialisation
    failed, _knowledge_service is None and the extension is not auto-installed.
    """
    defaults: list[tuple[str, bool]] = [
        ("github-integration", True),
        ("knowledge-search", _knowledge_service is not None),
    ]
    for ext_id, condition in defaults:
        if not condition:
            continue
        if _installed_extension_store is None or _extension_catalog is None or _package_installer is None:
            break
        if _installed_extension_store.is_installed(session, ext_id):
            continue
        pkg = _extension_catalog.get(ext_id)
        if pkg is None:
            print(
                f"[WARNING] Default extension {ext_id!r} not found in catalog; skipping.",
                file=sys.stderr,
            )
            continue
        permissions = [p.id for p in pkg.permissions]
        try:
            _package_installer.install(
                extension_id=ext_id,
                permissions_granted=permissions,
                session=session,
                auto_installed=True,
            )
        except Exception as exc:
            print(
                f"[WARNING] Auto-install of {ext_id!r} failed: {exc}",
                file=sys.stderr,
            )


async def run_startup_indexing() -> dict[str, int]:
    """Run knowledge indexing at startup. No-op if knowledge is not configured."""
    if _knowledge_indexer is None:
        return {}
    return await _index_with_summary(_knowledge_indexer)


async def _index_with_summary(indexer: KnowledgeIndexer) -> dict[str, int]:
    try:
        results = await indexer.index_all()
        total = sum(results.values())
        changed = sum(1 for c in results.values() if c > 0)
        print(
            f"[Knowledge] {len(results)} collection(s): "
            f"{changed} rebuilt, {total} chunk(s) total.",
            file=sys.stderr,
        )
        return results
    except Exception as exc:
        print(f"[WARNING] Knowledge indexing failed at startup: {exc}", file=sys.stderr)
        return {}


async def shutdown() -> None:
    """Close MCP connections and any other adapter resources. Called at app shutdown."""
    if _tool_registry is not None:
        for adapter in _tool_registry.get_all_adapters().values():
            await adapter.close()


def _create_llm_provider() -> ILLMProvider:
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIProvider()
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Set OPENAI_API_KEY in the environment or a .env file and restart the server."
    )


def get_orchestrator() -> Orchestrator:
    if _orchestrator is None:
        raise RuntimeError("Platform not initialized — call initialize() first")
    return _orchestrator


def get_run_manager() -> RunManager:
    if _run_manager is None:
        raise RuntimeError("Platform not initialized — call initialize() first")
    return _run_manager


def get_hitl_manager() -> ApprovalManager:
    if _hitl_manager is None:
        raise RuntimeError("Platform not initialized — call initialize() first")
    return _hitl_manager


def get_workflow_registry() -> WorkflowRegistry:
    if _workflow_registry is None:
        raise RuntimeError("Platform not initialized — call initialize() first")
    return _workflow_registry


def get_session_factory() -> sessionmaker[Session]:
    if _session_factory is None:
        raise RuntimeError("Platform not initialized — call initialize() first")
    return _session_factory


def get_knowledge_service() -> KnowledgeService | None:
    """Return the KnowledgeService, or None if knowledge is not configured."""
    return _knowledge_service


def get_db_session():
    """FastAPI dependency that yields a DB session per request."""
    if _session_factory is None:
        raise RuntimeError("Platform not initialized — call initialize() first")
    with _session_factory() as session:
        yield session


def get_capability_registry() -> CapabilityRegistry:
    if _capability_registry is None:
        raise RuntimeError("Platform not initialized — call initialize() first")
    return _capability_registry


def get_capability_manager() -> CapabilityManager:
    if _capability_manager is None:
        raise RuntimeError("Platform not initialized — call initialize() first")
    return _capability_manager


def get_planner_service() -> PlannerService:
    if _planner_service is None:
        raise RuntimeError("Platform not initialized — call initialize() first")
    return _planner_service


def get_execution_adapter() -> ExecutionAdapter:
    if _execution_adapter is None:
        raise RuntimeError("Platform not initialized — call initialize() first")
    return _execution_adapter


def get_extension_catalog() -> ExtensionCatalog:
    if _extension_catalog is None:
        raise RuntimeError("Platform not initialized — call initialize() first")
    return _extension_catalog


def get_installed_extension_store() -> InstalledExtensionStore:
    if _installed_extension_store is None:
        raise RuntimeError("Platform not initialized — call initialize() first")
    return _installed_extension_store


def get_package_installer() -> PackageInstaller:
    if _package_installer is None:
        raise RuntimeError("Platform not initialized — call initialize() first")
    return _package_installer

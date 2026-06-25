"""FastAPI dependency injection for platform services."""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from platform.config.loader import ConfigLoader
from platform.core.interfaces.llm import ILLMProvider
from platform.hitl.approval_manager import ApprovalManager
from platform.llm.openai_provider import OpenAIProvider
from platform.memory.in_memory_store import InMemoryStore
from platform.observability.composite_observer import CompositeObserver
from platform.observability.console_observer import ConsoleObserver
from platform.observability.persisting_observer import PersistingObserver
from platform.orchestrator.orchestrator import Orchestrator
from platform.orchestrator.run_manager import RunManager
from platform.persistence.database import Base, build_engine, build_session_factory
from platform.policy.engine import PolicyEngine
from platform.registries.agent_registry import AgentRegistry
from platform.registries.tool_registry import ToolRegistry
from platform.registries.workflow_registry import WorkflowRegistry
from platform.state.shared_state import SharedState

_orchestrator: Orchestrator | None = None
_run_manager: RunManager | None = None
_hitl_manager: ApprovalManager | None = None
_workflow_registry: WorkflowRegistry | None = None
_session_factory: sessionmaker[Session] | None = None


def initialize(workflows_dir: Path) -> None:
    """Create and wire all platform singletons. Called once from lifespan."""
    global _orchestrator, _run_manager, _hitl_manager, _workflow_registry, _session_factory

    wf_registry = WorkflowRegistry()
    ag_registry = AgentRegistry()
    tl_registry = ToolRegistry()
    ConfigLoader(wf_registry, ag_registry, tl_registry).load_all(workflows_dir)

    database_url = os.environ.get("DATABASE_URL", "sqlite:///./workflow.db")
    engine = build_engine(database_url)
    Base.metadata.create_all(engine)
    session_factory = build_session_factory(engine)
    _session_factory = session_factory

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


def get_db_session():
    """FastAPI dependency that yields a DB session per request."""
    if _session_factory is None:
        raise RuntimeError("Platform not initialized — call initialize() first")
    with _session_factory() as session:
        yield session

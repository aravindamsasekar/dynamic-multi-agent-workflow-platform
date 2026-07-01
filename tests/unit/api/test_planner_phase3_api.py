"""Phase 3 planner API tests: pending_install state, install suggestions, regeneration.

Uses a test-only FastAPI app with dependency overrides and in-memory SQLite.
No OPENAI_API_KEY or real filesystem required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.dependencies import (
    get_capability_manager,
    get_db_session,
    get_execution_adapter,
    get_package_installer,
    get_planner_service,
)
from api.routers import planner as planner_router
from platform.extensions.catalog import ExtensionCatalog
from platform.extensions.installer import PackageInstaller
from platform.extensions.manager import CapabilityManager
from platform.extensions.models import InstallSuggestion, PermissionSummary
from platform.persistence.database import Base
from platform.persistence.repositories.package_repo import InstalledExtensionStore
from platform.persistence.repositories.plan_repo import PlanRepository
from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.models import (
    GeneratedWorkflowPlan,
    GoalAnalysis,
    GuardrailConfig,
    RiskLevel,
    RuntimeAgentDefinition,
    ValidationError,
    ValidationResult,
)
from platform.registries.tool_registry import ToolRegistry

_EXTENSIONS_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "resources" / "extensions"
)
_CATALOG = ExtensionCatalog.load(_EXTENSIONS_DIR)

_PLAN_ID = "phase3-plan-001"

_INSTALL_SUGGESTION = InstallSuggestion(
    extension_id="filesystem-reader",
    name="Filesystem Reader",
    description="Read local files and return their UTF-8 contents.",
    capabilities_provided=["filesystem_read"],
    permissions=[PermissionSummary(id="read_local_files_readonly", risk_level="low")],
)


def _make_pending_install_plan(plan_id: str = _PLAN_ID) -> GeneratedWorkflowPlan:
    """A non-executable plan missing filesystem_read capability."""
    return GeneratedWorkflowPlan(
        plan_id=plan_id,
        user_goal="Read the contents of config.yaml",
        goal_analysis=GoalAnalysis(
            required_capabilities=["filesystem_read"],
            risk_level=RiskLevel.LOW,
            confidence=0.85,
            reasoning="Needs filesystem access.",
            constraints=[],
            requires_hitl=False,
        ),
        selected_pattern="",
        selected_agents=[],
        selected_tools=[],
        guardrails=[],
        hitl_required=False,
        warnings=[],
        explanation="Missing capabilities: ['filesystem_read']. Install the suggested extension(s).",
        estimated_complexity="unknown",
        estimated_duration_seconds=0,
        task_label="",
        executable=False,
        missing_capabilities=["filesystem_read"],
        install_suggestions=[_INSTALL_SUGGESTION],
        unsupported=False,
    )


def _make_valid_plan(plan_id: str = _PLAN_ID) -> GeneratedWorkflowPlan:
    """A fully executable plan for the same goal (post-install)."""
    return GeneratedWorkflowPlan(
        plan_id=plan_id,
        user_goal="Read the contents of config.yaml",
        goal_analysis=GoalAnalysis(
            required_capabilities=["filesystem_read"],
            risk_level=RiskLevel.LOW,
            confidence=0.9,
            reasoning="Filesystem tool available.",
            constraints=[],
            requires_hitl=False,
        ),
        selected_pattern="single_agent",
        selected_agents=["filesystem_agent"],
        selected_tools=["filesystem_read_file"],
        guardrails=[],
        hitl_required=False,
        warnings=[],
        explanation="Reads a file using the filesystem agent.",
        estimated_complexity="low",
        estimated_duration_seconds=5,
        task_label="filesystem",
        executable=True,
    )


def _make_invalid_validation() -> ValidationResult:
    return ValidationResult(
        is_valid=False,
        errors=[ValidationError(code="MISSING_CAPABILITIES", message="Missing: ['filesystem_read']")],
        warnings=[],
    )


def _make_valid_validation() -> ValidationResult:
    return ValidationResult(is_valid=True, errors=[], warnings=[])


# ---------------------------------------------------------------------------
# Mock services
# ---------------------------------------------------------------------------


class _PendingInstallPlannerService:
    """Planner that always returns a pending_install plan."""
    async def generate(self, goal: str):
        return _make_pending_install_plan(), _make_invalid_validation()


class _ExecutablePlannerService:
    """Planner that always returns a fully executable plan."""
    async def generate(self, goal: str):
        return _make_valid_plan(), _make_valid_validation()


class _MockInstaller:
    """Installer that records calls but never touches a DB."""
    def __init__(self):
        self.installed: list[str] = []

    def install(self, extension_id, permissions_granted, session):
        self.installed.append(extension_id)

    def restore_from_db(self, session):
        pass


class _TransitioningPlannerService:
    """Returns pending_install on first call, executable on second."""
    def __init__(self):
        self._calls = 0

    async def generate(self, goal: str):
        self._calls += 1
        if self._calls == 1:
            return _make_pending_install_plan(), _make_invalid_validation()
        return _make_valid_plan(), _make_valid_validation()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_factory():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _build_app(planner_svc, installer=None, db_factory_=None):
    """Build a test FastAPI app with the given planner service."""
    mock_installer = installer or _MockInstaller()
    cap_mgr = CapabilityManager(_CATALOG, CapabilityRegistry())

    app = FastAPI()
    app.include_router(planner_router.router, prefix="/planner", tags=["planner"])
    app.dependency_overrides[get_planner_service] = lambda: planner_svc
    app.dependency_overrides[get_package_installer] = lambda: mock_installer
    app.dependency_overrides[get_capability_manager] = lambda: cap_mgr

    if db_factory_ is not None:
        def _session():
            with db_factory_() as s:
                yield s
        app.dependency_overrides[get_db_session] = _session

    return app


# ---------------------------------------------------------------------------
# POST /planner/generate — pending_install path
# ---------------------------------------------------------------------------


class TestGeneratePlanPendingInstall:
    async def test_status_is_pending_install_when_capabilities_missing(self, db_factory) -> None:
        app = _build_app(_PendingInstallPlannerService(), db_factory_=db_factory)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/planner/generate", json={"goal": "Read config.yaml"})
        assert response.status_code == 201
        assert response.json()["status"] == "pending_install"

    async def test_executable_is_false_for_pending_install(self, db_factory) -> None:
        app = _build_app(_PendingInstallPlannerService(), db_factory_=db_factory)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/planner/generate", json={"goal": "Read config.yaml"})
        assert response.json()["executable"] is False

    async def test_missing_capabilities_populated(self, db_factory) -> None:
        app = _build_app(_PendingInstallPlannerService(), db_factory_=db_factory)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/planner/generate", json={"goal": "Read config.yaml"})
        assert "filesystem_read" in response.json()["missing_capabilities"]

    async def test_install_suggestions_populated(self, db_factory) -> None:
        app = _build_app(_PendingInstallPlannerService(), db_factory_=db_factory)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/planner/generate", json={"goal": "Read config.yaml"})
        suggestions = response.json()["install_suggestions"]
        assert len(suggestions) >= 1
        assert suggestions[0]["extension_id"] == "filesystem-reader"

    async def test_suggestion_has_permission_fields(self, db_factory) -> None:
        app = _build_app(_PendingInstallPlannerService(), db_factory_=db_factory)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/planner/generate", json={"goal": "Read config.yaml"})
        perm = response.json()["install_suggestions"][0]["permissions"][0]
        assert "id" in perm
        assert "risk_level" in perm

    async def test_status_is_pending_review_when_all_satisfied(self, db_factory) -> None:
        app = _build_app(_ExecutablePlannerService(), db_factory_=db_factory)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/planner/generate", json={"goal": "Read config.yaml"})
        assert response.json()["status"] == "pending_review"

    async def test_executable_true_when_all_satisfied(self, db_factory) -> None:
        app = _build_app(_ExecutablePlannerService(), db_factory_=db_factory)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/planner/generate", json={"goal": "Read config.yaml"})
        assert response.json()["executable"] is True

    async def test_unsupported_false_for_installable_missing(self, db_factory) -> None:
        app = _build_app(_PendingInstallPlannerService(), db_factory_=db_factory)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/planner/generate", json={"goal": "Read config.yaml"})
        assert response.json()["unsupported"] is False


# ---------------------------------------------------------------------------
# POST /planner/{plan_id}/install
# ---------------------------------------------------------------------------


class TestInstallAndRegenerate:
    async def test_404_when_plan_not_found(self, db_factory) -> None:
        app = _build_app(_ExecutablePlannerService(), db_factory_=db_factory)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/planner/no-such-plan/install")
        assert response.status_code == 404

    async def test_409_when_plan_not_in_pending_install(self, db_factory) -> None:
        repo = PlanRepository()
        with db_factory() as s:
            repo.create(s, _make_valid_plan(), _make_valid_validation(), status="pending_review")
            s.commit()

        app = _build_app(_ExecutablePlannerService(), db_factory_=db_factory)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(f"/planner/{_PLAN_ID}/install")
        assert response.status_code == 409

    async def test_returns_200_on_success(self, db_factory) -> None:
        repo = PlanRepository()
        with db_factory() as s:
            repo.create(s, _make_pending_install_plan(), _make_invalid_validation(), status="pending_install")
            s.commit()

        app = _build_app(_ExecutablePlannerService(), db_factory_=db_factory)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(f"/planner/{_PLAN_ID}/install")
        assert response.status_code == 200

    async def test_preserves_plan_id_after_regeneration(self, db_factory) -> None:
        repo = PlanRepository()
        with db_factory() as s:
            repo.create(s, _make_pending_install_plan(), _make_invalid_validation(), status="pending_install")
            s.commit()

        app = _build_app(_ExecutablePlannerService(), db_factory_=db_factory)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(f"/planner/{_PLAN_ID}/install")
        assert response.json()["plan_id"] == _PLAN_ID

    async def test_status_transitions_to_pending_review(self, db_factory) -> None:
        repo = PlanRepository()
        with db_factory() as s:
            repo.create(s, _make_pending_install_plan(), _make_invalid_validation(), status="pending_install")
            s.commit()

        app = _build_app(_ExecutablePlannerService(), db_factory_=db_factory)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(f"/planner/{_PLAN_ID}/install")
        assert response.json()["status"] == "pending_review"

    async def test_executable_true_after_successful_install(self, db_factory) -> None:
        repo = PlanRepository()
        with db_factory() as s:
            repo.create(s, _make_pending_install_plan(), _make_invalid_validation(), status="pending_install")
            s.commit()

        app = _build_app(_ExecutablePlannerService(), db_factory_=db_factory)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(f"/planner/{_PLAN_ID}/install")
        assert response.json()["executable"] is True

    async def test_installer_called_for_each_suggestion(self, db_factory) -> None:
        repo = PlanRepository()
        with db_factory() as s:
            repo.create(s, _make_pending_install_plan(), _make_invalid_validation(), status="pending_install")
            s.commit()

        installer = _MockInstaller()
        app = _build_app(_ExecutablePlannerService(), installer=installer, db_factory_=db_factory)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(f"/planner/{_PLAN_ID}/install")
        assert "filesystem-reader" in installer.installed

    async def test_db_row_reflects_new_plan_content(self, db_factory) -> None:
        repo = PlanRepository()
        with db_factory() as s:
            repo.create(s, _make_pending_install_plan(), _make_invalid_validation(), status="pending_install")
            s.commit()

        app = _build_app(_ExecutablePlannerService(), db_factory_=db_factory)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(f"/planner/{_PLAN_ID}/install")

        with db_factory() as s:
            row = repo.get(s, _PLAN_ID)
        assert row is not None
        assert row.status == "pending_review"

    async def test_get_plan_returns_updated_plan_after_install(self, db_factory) -> None:
        repo = PlanRepository()
        with db_factory() as s:
            repo.create(s, _make_pending_install_plan(), _make_invalid_validation(), status="pending_install")
            s.commit()

        app = _build_app(_ExecutablePlannerService(), db_factory_=db_factory)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(f"/planner/{_PLAN_ID}/install")
            get_resp = await client.get(f"/planner/{_PLAN_ID}")
        assert get_resp.status_code == 200
        assert get_resp.json()["executable"] is True
        assert get_resp.json()["status"] == "pending_review"

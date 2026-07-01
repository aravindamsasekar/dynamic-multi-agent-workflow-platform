"""Unit tests for planner API endpoints.

Uses a test-only FastAPI app (no lifespan) with dependency overrides so that
no OPENAI_API_KEY is required and all I/O uses in-memory SQLite.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.dependencies import (
    get_db_session,
    get_execution_adapter,
    get_planner_service,
)
from api.routers import planner as planner_router
from platform.core.models.workflow import RunStatus, WorkflowResult
from platform.persistence.database import Base
from platform.persistence.repositories.plan_repo import PlanRepository
from platform.planner.models import (
    GeneratedWorkflowPlan,
    GoalAnalysis,
    GuardrailConfig,
    RiskLevel,
    ValidationError,
    ValidationResult,
    ValidationWarning,
)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_PLAN_ID = "test-plan-001"


def _make_plan(plan_id: str = _PLAN_ID) -> GeneratedWorkflowPlan:
    return GeneratedWorkflowPlan(
        plan_id=plan_id,
        user_goal="Review PR #42",
        goal_analysis=GoalAnalysis(
            required_capabilities=["fetch_pr_data", "synthesize_findings"],
            risk_level=RiskLevel.LOW,
            confidence=0.9,
            reasoning="Looks like a code review.",
            constraints=["read_only"],
            requires_hitl=False,
        ),
        selected_pattern="parallel_specialist",
        selected_agents=["pr_data_agent", "review_specialist", "synthesis_agent"],
        selected_tools=["github_get_pr", "github_get_diff", "knowledge_search"],
        guardrails=[GuardrailConfig(rule_type="content_filter", config={}, reason="safety")],
        hitl_required=False,
        warnings=[],
        explanation="A parallel PR review workflow.",
        estimated_complexity="medium",
        estimated_duration_seconds=65,
    )


def _make_validation(is_valid: bool = True) -> ValidationResult:
    return ValidationResult(
        is_valid=is_valid,
        errors=[] if is_valid else [ValidationError(code="MISSING_AGENT", message="Missing.")],
        warnings=[ValidationWarning(code="LOW_CONFIDENCE", message="Confidence borderline.")],
    )


# ---------------------------------------------------------------------------
# Mock services
# ---------------------------------------------------------------------------


class _MockPlannerService:
    async def generate(self, goal: str):
        return _make_plan(), _make_validation()


class _MockExecutionAdapter:
    async def execute(self, plan, input_data):
        return WorkflowResult(
            run_id="run-from-adapter",
            workflow_id=plan.plan_id,
            output="Review complete.",
            status=RunStatus.COMPLETED,
        )


# ---------------------------------------------------------------------------
# Test app (no lifespan)
# ---------------------------------------------------------------------------

_test_app = FastAPI()
_test_app.include_router(planner_router.router, prefix="/planner", tags=["planner"])
_test_app.dependency_overrides[get_planner_service] = lambda: _MockPlannerService()
_test_app.dependency_overrides[get_execution_adapter] = lambda: _MockExecutionAdapter()


# ---------------------------------------------------------------------------
# DB fixtures
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


@pytest.fixture
def seeded_db(db_factory):
    """DB pre-loaded with one pending plan."""
    repo = PlanRepository()
    with db_factory() as s:
        repo.create(s, _make_plan(), _make_validation())
        s.commit()
    return db_factory


@pytest.fixture
def empty_db(db_factory):
    return db_factory


def _session_override(factory):
    def _inner():
        with factory() as s:
            yield s
    return _inner


# ---------------------------------------------------------------------------
# Shared client fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def client(seeded_db):
    _test_app.dependency_overrides[get_db_session] = _session_override(seeded_db)
    async with AsyncClient(transport=ASGITransport(app=_test_app), base_url="http://test") as c:
        yield c
    _test_app.dependency_overrides.pop(get_db_session, None)


@pytest.fixture
async def empty_client(empty_db):
    _test_app.dependency_overrides[get_db_session] = _session_override(empty_db)
    async with AsyncClient(transport=ASGITransport(app=_test_app), base_url="http://test") as c:
        yield c
    _test_app.dependency_overrides.pop(get_db_session, None)


# ---------------------------------------------------------------------------
# POST /planner/generate
# ---------------------------------------------------------------------------


class TestGeneratePlan:
    async def test_returns_201(self, empty_client: AsyncClient):
        response = await empty_client.post("/planner/generate", json={"goal": "Review my PR"})
        assert response.status_code == 201

    async def test_response_contains_plan_id(self, empty_client: AsyncClient):
        response = await empty_client.post("/planner/generate", json={"goal": "Review my PR"})
        data = response.json()
        assert "plan_id" in data
        assert data["plan_id"] == _PLAN_ID

    async def test_response_contains_goal(self, empty_client: AsyncClient):
        response = await empty_client.post("/planner/generate", json={"goal": "Review my PR"})
        data = response.json()
        assert data["goal"] == "Review PR #42"

    async def test_response_status_is_pending_review(self, empty_client: AsyncClient):
        response = await empty_client.post("/planner/generate", json={"goal": "Review my PR"})
        data = response.json()
        assert data["status"] == "pending_review"

    async def test_response_includes_selected_pattern(self, empty_client: AsyncClient):
        response = await empty_client.post("/planner/generate", json={"goal": "Review PR"})
        data = response.json()
        assert data["selected_pattern"] == "parallel_specialist"

    async def test_response_includes_selected_agents(self, empty_client: AsyncClient):
        response = await empty_client.post("/planner/generate", json={"goal": "Review PR"})
        data = response.json()
        assert len(data["selected_agents"]) > 0

    async def test_response_includes_validation(self, empty_client: AsyncClient):
        response = await empty_client.post("/planner/generate", json={"goal": "Review PR"})
        data = response.json()
        assert "validation" in data
        assert data["validation"]["is_valid"] is True

    async def test_response_includes_complexity(self, empty_client: AsyncClient):
        response = await empty_client.post("/planner/generate", json={"goal": "Review PR"})
        data = response.json()
        assert data["estimated_complexity"] == "medium"

    async def test_response_includes_duration(self, empty_client: AsyncClient):
        response = await empty_client.post("/planner/generate", json={"goal": "Review PR"})
        data = response.json()
        assert data["estimated_duration_seconds"] == 65

    async def test_response_includes_goal_analysis(self, empty_client: AsyncClient):
        response = await empty_client.post("/planner/generate", json={"goal": "Review PR"})
        data = response.json()
        ga = data["goal_analysis"]
        assert "task_type" not in ga
        assert ga["risk_level"] == "low"

    async def test_response_includes_task_label(self, empty_client: AsyncClient):
        response = await empty_client.post("/planner/generate", json={"goal": "Review PR"})
        data = response.json()
        assert "task_label" in data

    async def test_response_includes_explanation(self, empty_client: AsyncClient):
        response = await empty_client.post("/planner/generate", json={"goal": "Review PR"})
        data = response.json()
        assert data["explanation"] == "A parallel PR review workflow."

    async def test_planner_error_returns_422(self, empty_db):
        from platform.planner.models import PlannerError

        class _FailingPlanner:
            async def generate(self, goal: str):
                raise PlannerError("LLM call failed")

        _test_app.dependency_overrides[get_planner_service] = lambda: _FailingPlanner()
        _test_app.dependency_overrides[get_db_session] = _session_override(empty_db)
        try:
            async with AsyncClient(transport=ASGITransport(app=_test_app), base_url="http://test") as c:
                response = await c.post("/planner/generate", json={"goal": "Review PR"})
            assert response.status_code == 422
            assert "LLM call failed" in response.json()["detail"]
        finally:
            _test_app.dependency_overrides[get_planner_service] = lambda: _MockPlannerService()
            _test_app.dependency_overrides.pop(get_db_session, None)


# ---------------------------------------------------------------------------
# GET /planner/{plan_id}
# ---------------------------------------------------------------------------


class TestGetPlan:
    async def test_returns_200_for_existing_plan(self, client: AsyncClient):
        response = await client.get(f"/planner/{_PLAN_ID}")
        assert response.status_code == 200

    async def test_returns_404_for_missing_plan(self, client: AsyncClient):
        response = await client.get("/planner/nonexistent-plan-id")
        assert response.status_code == 404

    async def test_response_plan_id_matches(self, client: AsyncClient):
        response = await client.get(f"/planner/{_PLAN_ID}")
        data = response.json()
        assert data["plan_id"] == _PLAN_ID

    async def test_response_includes_validation(self, client: AsyncClient):
        response = await client.get(f"/planner/{_PLAN_ID}")
        data = response.json()
        assert "validation" in data

    async def test_response_status_from_db(self, client: AsyncClient):
        response = await client.get(f"/planner/{_PLAN_ID}")
        data = response.json()
        assert data["status"] == "pending_review"

    async def test_response_goal_matches(self, client: AsyncClient):
        response = await client.get(f"/planner/{_PLAN_ID}")
        data = response.json()
        assert data["goal"] == "Review PR #42"


# ---------------------------------------------------------------------------
# POST /planner/{plan_id}/approve
# ---------------------------------------------------------------------------


class TestApprovePlan:
    async def test_returns_200_for_pending_plan(self, client: AsyncClient):
        response = await client.post(f"/planner/{_PLAN_ID}/approve", json={"input_data": "pr info"})
        assert response.status_code == 200

    async def test_response_contains_run_id(self, client: AsyncClient):
        response = await client.post(f"/planner/{_PLAN_ID}/approve", json={"input_data": "pr info"})
        data = response.json()
        assert data["run_id"] == "run-from-adapter"

    async def test_response_contains_plan_id(self, client: AsyncClient):
        response = await client.post(f"/planner/{_PLAN_ID}/approve", json={"input_data": "pr info"})
        data = response.json()
        assert data["plan_id"] == _PLAN_ID

    async def test_response_status_is_completed(self, client: AsyncClient):
        response = await client.post(f"/planner/{_PLAN_ID}/approve", json={"input_data": "pr info"})
        data = response.json()
        assert data["status"] == "completed"

    async def test_response_includes_output(self, client: AsyncClient):
        response = await client.post(f"/planner/{_PLAN_ID}/approve", json={"input_data": "pr info"})
        data = response.json()
        assert data["output"] == "Review complete."

    async def test_returns_404_for_missing_plan(self, client: AsyncClient):
        response = await client.post("/planner/nonexistent/approve", json={"input_data": ""})
        assert response.status_code == 404

    async def test_returns_409_for_already_executed_plan(self, seeded_db):
        repo = PlanRepository()
        with seeded_db() as s:
            repo.update_status(s, _PLAN_ID, "executed")
            s.commit()

        _test_app.dependency_overrides[get_db_session] = _session_override(seeded_db)
        try:
            async with AsyncClient(transport=ASGITransport(app=_test_app), base_url="http://test") as c:
                response = await c.post(f"/planner/{_PLAN_ID}/approve", json={"input_data": ""})
            assert response.status_code == 409
        finally:
            _test_app.dependency_overrides.pop(get_db_session, None)

    async def test_approve_accepts_empty_input_data(self, client: AsyncClient):
        response = await client.post(f"/planner/{_PLAN_ID}/approve", json={})
        assert response.status_code == 200

    async def test_approve_accepts_dict_input(self, client: AsyncClient):
        input_data = {"owner": "org", "repo": "myrepo", "pr_number": 42}
        response = await client.post(f"/planner/{_PLAN_ID}/approve", json={"input_data": input_data})
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# POST /planner/{plan_id}/reject
# ---------------------------------------------------------------------------


class TestRejectPlan:
    async def test_returns_200_for_pending_plan(self, client: AsyncClient):
        response = await client.post(f"/planner/{_PLAN_ID}/reject", json={"reason": "Not needed."})
        assert response.status_code == 200

    async def test_response_status_is_rejected(self, client: AsyncClient):
        response = await client.post(f"/planner/{_PLAN_ID}/reject", json={"reason": "Not needed."})
        data = response.json()
        assert data["status"] == "rejected"

    async def test_response_plan_id_matches(self, client: AsyncClient):
        response = await client.post(f"/planner/{_PLAN_ID}/reject", json={"reason": "Cancel."})
        data = response.json()
        assert data["plan_id"] == _PLAN_ID

    async def test_response_goal_matches(self, client: AsyncClient):
        response = await client.post(f"/planner/{_PLAN_ID}/reject", json={})
        data = response.json()
        assert data["goal"] == "Review PR #42"

    async def test_response_includes_timestamps(self, client: AsyncClient):
        response = await client.post(f"/planner/{_PLAN_ID}/reject", json={})
        data = response.json()
        assert "created_at" in data
        assert "updated_at" in data

    async def test_returns_404_for_missing_plan(self, client: AsyncClient):
        response = await client.post("/planner/no-such-plan/reject", json={})
        assert response.status_code == 404

    async def test_returns_409_for_already_rejected(self, seeded_db):
        repo = PlanRepository()
        with seeded_db() as s:
            repo.update_status(s, _PLAN_ID, "rejected")
            s.commit()

        _test_app.dependency_overrides[get_db_session] = _session_override(seeded_db)
        try:
            async with AsyncClient(transport=ASGITransport(app=_test_app), base_url="http://test") as c:
                response = await c.post(f"/planner/{_PLAN_ID}/reject", json={})
            assert response.status_code == 409
        finally:
            _test_app.dependency_overrides.pop(get_db_session, None)

    async def test_reject_no_body_uses_default_reason(self, client: AsyncClient):
        response = await client.post(f"/planner/{_PLAN_ID}/reject", json={})
        assert response.status_code == 200

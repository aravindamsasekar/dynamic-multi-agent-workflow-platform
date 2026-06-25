"""Unit tests for FastAPI endpoints.

Uses a test-only FastAPI app (no lifespan) with dependency overrides so that
no OPENAI_API_KEY and no workflow YAML files are needed.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.dependencies import get_db_session, get_hitl_manager, get_orchestrator, get_run_manager, get_workflow_registry
from api.routers import hitl as hitl_router
from api.routers import runs as runs_router
from api.routers import workflows as workflows_router
from platform.core.exceptions import RunNotFound, WorkflowNotFound
from platform.core.models.workflow import (
    PatternType,
    RunStatus,
    WorkflowDefinition,
    WorkflowResult,
    WorkflowRun,
)
from platform.persistence.database import Base
from platform.persistence.repositories.agent_repo import AgentRepository
from platform.persistence.repositories.event_repo import EventRepository
from platform.persistence.repositories.run_repo import RunRepository
from platform.persistence.repositories.tool_repo import ToolRepository

_TS = datetime(2024, 1, 15, 10, 0, 0)

# ---------------------------------------------------------------------------
# Mock services
# ---------------------------------------------------------------------------


_TEST_WF = WorkflowDefinition(
    workflow_id="incident_commander",
    name="Incident Commander",
    description="Parallel specialist demo workflow",
    pattern=PatternType.PARALLEL_SPECIALIST,
    hitl_enabled=False,
)

_TEST_RUN = WorkflowRun(
    run_id="test-run-001",
    workflow_id="incident_commander",
    status=RunStatus.COMPLETED,
    input="Production alert",
    output="Root cause: DB pool exhaustion.",
)

_TEST_RESULT = WorkflowResult(
    run_id="test-run-001",
    workflow_id="incident_commander",
    output="Root cause: DB pool exhaustion.",
)


class _MockWorkflowRegistry:
    def get(self, workflow_id: str) -> WorkflowDefinition:
        if workflow_id == "incident_commander":
            return _TEST_WF
        raise WorkflowNotFound(f"Workflow '{workflow_id}' not found")

    def list_all(self) -> list[WorkflowDefinition]:
        return [_TEST_WF]


class _MockOrchestrator:
    async def run(self, workflow_id: str, input: str) -> WorkflowResult:
        if workflow_id != "incident_commander":
            raise WorkflowNotFound(f"Workflow '{workflow_id}' not found")
        return _TEST_RESULT


class _MockRunManager:
    def get_run(self, run_id: str) -> WorkflowRun:
        if run_id == "test-run-001":
            return _TEST_RUN
        raise RunNotFound(f"Run '{run_id}' not found")


# ---------------------------------------------------------------------------
# In-memory DB fixture shared by API tests
# ---------------------------------------------------------------------------


@pytest.fixture
def db_session_factory():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


@pytest.fixture
def seeded_db(db_session_factory):
    """Seed one run with agent, tool, and event rows."""
    run_repo = RunRepository()
    agent_repo = AgentRepository()
    tool_repo = ToolRepository()
    event_repo = EventRepository()

    with db_session_factory() as s:
        run_repo.create(s, "test-run-001", "incident_commander", "completed", "Production alert", _TS, _TS)
        agent_repo.create(s, "test-run-001", "metrics_agent", "CPU high", _TS)
        tool_repo.create(s, "test-run-001", "mock_tool", {"k": "v"}, "ok", False, _TS)
        event_repo.create(s, "test-run-001", "workflow_started", {"workflow_id": "incident_commander"}, _TS)
        event_repo.create(s, "test-run-001", "workflow_completed", {"output": "done"}, _TS)
        s.commit()
    return db_session_factory


def _make_db_session_override(factory):
    def _override():
        with factory() as s:
            yield s
    return _override


# ---------------------------------------------------------------------------
# Test app — no lifespan, shares same routers as api.main
# ---------------------------------------------------------------------------

_test_app = FastAPI()


@_test_app.exception_handler(WorkflowNotFound)
async def _wnf(request, exc: WorkflowNotFound):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@_test_app.exception_handler(RunNotFound)
async def _rnf(request, exc: RunNotFound):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


_test_app.include_router(workflows_router.router, prefix="/workflows", tags=["workflows"])
_test_app.include_router(runs_router.router, prefix="/runs", tags=["runs"])
_test_app.include_router(hitl_router.router, prefix="/runs", tags=["hitl"])

_test_app.dependency_overrides[get_workflow_registry] = lambda: _MockWorkflowRegistry()
_test_app.dependency_overrides[get_orchestrator] = lambda: _MockOrchestrator()
_test_app.dependency_overrides[get_run_manager] = lambda: _MockRunManager()
_test_app.dependency_overrides[get_hitl_manager] = lambda: None


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
async def client(seeded_db):
    _test_app.dependency_overrides[get_db_session] = _make_db_session_override(seeded_db)
    async with AsyncClient(
        transport=ASGITransport(app=_test_app), base_url="http://test"
    ) as c:
        yield c
    _test_app.dependency_overrides.pop(get_db_session, None)


# ---------------------------------------------------------------------------
# Workflow endpoint tests
# ---------------------------------------------------------------------------


class TestWorkflowEndpoints:
    async def test_list_workflows(self, client: AsyncClient) -> None:
        response = await client.get("/workflows/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["workflow_id"] == "incident_commander"

    async def test_get_workflow_found(self, client: AsyncClient) -> None:
        response = await client.get("/workflows/incident_commander")
        assert response.status_code == 200
        data = response.json()
        assert data["workflow_id"] == "incident_commander"
        assert data["pattern"] == "parallel_specialist"
        assert data["hitl_enabled"] is False

    async def test_get_workflow_not_found(self, client: AsyncClient) -> None:
        response = await client.get("/workflows/nonexistent")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Run endpoint tests
# ---------------------------------------------------------------------------


class TestRunEndpoints:
    async def test_create_run_success(self, client: AsyncClient) -> None:
        response = await client.post(
            "/runs/",
            json={"workflow_id": "incident_commander", "input": "Production alert"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == "test-run-001"
        assert data["workflow_id"] == "incident_commander"
        assert data["output"] == "Root cause: DB pool exhaustion."

    async def test_create_run_workflow_not_found(self, client: AsyncClient) -> None:
        response = await client.post(
            "/runs/",
            json={"workflow_id": "nonexistent", "input": "test"},
        )
        assert response.status_code == 404

    async def test_get_run_found(self, client: AsyncClient) -> None:
        response = await client.get("/runs/test-run-001")
        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == "test-run-001"
        assert data["status"] == "completed"
        assert data["output"] == "Root cause: DB pool exhaustion."

    async def test_get_run_not_found(self, client: AsyncClient) -> None:
        response = await client.get("/runs/nonexistent-run")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /runs — list endpoint
# ---------------------------------------------------------------------------


class TestListRunsEndpoint:
    async def test_list_runs_returns_list(self, client: AsyncClient) -> None:
        response = await client.get("/runs/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["run_id"] == "test-run-001"
        assert data[0]["workflow_id"] == "incident_commander"
        assert data[0]["status"] == "completed"

    async def test_list_runs_includes_timestamps(self, client: AsyncClient) -> None:
        response = await client.get("/runs/")
        data = response.json()
        assert "created_at" in data[0]
        assert "updated_at" in data[0]


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/events
# ---------------------------------------------------------------------------


class TestRunEventsEndpoint:
    async def test_events_returns_list(self, client: AsyncClient) -> None:
        response = await client.get("/runs/test-run-001/events")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["event_type"] == "workflow_started"
        assert data[1]["event_type"] == "workflow_completed"

    async def test_events_includes_payload(self, client: AsyncClient) -> None:
        response = await client.get("/runs/test-run-001/events")
        data = response.json()
        assert data[0]["payload"] is not None
        assert "workflow_id" in data[0]["payload"]

    async def test_events_not_found(self, client: AsyncClient) -> None:
        response = await client.get("/runs/ghost-run/events")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/details
# ---------------------------------------------------------------------------


class TestRunDetailsEndpoint:
    async def test_details_returns_run_fields(self, client: AsyncClient) -> None:
        response = await client.get("/runs/test-run-001/details")
        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == "test-run-001"
        assert data["workflow_id"] == "incident_commander"
        assert data["status"] == "completed"
        assert data["input"] == "Production alert"

    async def test_details_includes_agent_results(self, client: AsyncClient) -> None:
        response = await client.get("/runs/test-run-001/details")
        data = response.json()
        assert len(data["agent_results"]) == 1
        assert data["agent_results"][0]["agent_id"] == "metrics_agent"
        assert data["agent_results"][0]["output"] == "CPU high"

    async def test_details_includes_tool_calls(self, client: AsyncClient) -> None:
        response = await client.get("/runs/test-run-001/details")
        data = response.json()
        assert len(data["tool_calls"]) == 1
        assert data["tool_calls"][0]["tool_name"] == "mock_tool"
        assert data["tool_calls"][0]["output"] == "ok"
        assert data["tool_calls"][0]["is_error"] is False

    async def test_details_includes_events(self, client: AsyncClient) -> None:
        response = await client.get("/runs/test-run-001/details")
        data = response.json()
        assert len(data["events"]) == 2

    async def test_details_not_found(self, client: AsyncClient) -> None:
        response = await client.get("/runs/ghost-run/details")
        assert response.status_code == 404

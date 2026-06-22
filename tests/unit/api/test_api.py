"""Unit tests for FastAPI endpoints.

Uses a test-only FastAPI app (no lifespan) with dependency overrides so that
no OPENAI_API_KEY and no workflow YAML files are needed.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from api.dependencies import get_hitl_manager, get_orchestrator, get_run_manager, get_workflow_registry
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
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=_test_app), base_url="http://test"
    ) as c:
        yield c


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

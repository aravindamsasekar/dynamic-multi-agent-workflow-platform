"""Workflow run trigger and status endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_orchestrator, get_run_manager
from api.schemas.run import RunRequest, RunStatusResponse
from platform.orchestrator.orchestrator import Orchestrator
from platform.orchestrator.run_manager import RunManager

router = APIRouter()


@router.post("/", response_model=RunStatusResponse)
async def create_run(
    request: RunRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    run_manager: RunManager = Depends(get_run_manager),
) -> RunStatusResponse:
    result = await orchestrator.run(request.workflow_id, request.input)
    run = run_manager.get_run(result.run_id)
    return RunStatusResponse(
        run_id=run.run_id,
        workflow_id=run.workflow_id,
        status=run.status.value,
        output=run.output,
    )


@router.get("/{run_id}", response_model=RunStatusResponse)
async def get_run(
    run_id: str,
    run_manager: RunManager = Depends(get_run_manager),
) -> RunStatusResponse:
    run = run_manager.get_run(run_id)
    return RunStatusResponse(
        run_id=run.run_id,
        workflow_id=run.workflow_id,
        status=run.status.value,
        output=run.output,
        error=run.error,
    )

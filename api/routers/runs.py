"""Workflow run trigger and status endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import get_db_session, get_orchestrator, get_run_manager
from api.schemas.run import (
    AgentResultResponse,
    EventResponse,
    RunDetailsResponse,
    RunListItemResponse,
    RunRequest,
    RunStatusResponse,
    ToolCallResponse,
)
from platform.orchestrator.orchestrator import Orchestrator
from platform.orchestrator.run_manager import RunManager
from platform.persistence.repositories.agent_repo import AgentRepository
from platform.persistence.repositories.event_repo import EventRepository
from platform.persistence.repositories.run_repo import RunRepository
from platform.persistence.repositories.tool_repo import ToolRepository

router = APIRouter()

_run_repo = RunRepository()
_agent_repo = AgentRepository()
_tool_repo = ToolRepository()
_event_repo = EventRepository()


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


@router.get("/", response_model=list[RunListItemResponse])
def list_runs(
    session: Session = Depends(get_db_session),
) -> list[RunListItemResponse]:
    rows = _run_repo.list_all(session)
    return [
        RunListItemResponse(
            run_id=r.run_id,
            workflow_id=r.workflow_id,
            status=r.status,
            created_at=r.created_at,
            updated_at=r.updated_at,
        )
        for r in rows
    ]


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


@router.get("/{run_id}/events", response_model=list[EventResponse])
def list_run_events(
    run_id: str,
    session: Session = Depends(get_db_session),
) -> list[EventResponse]:
    run_row = _run_repo.get(session, run_id)
    if run_row is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    rows = _event_repo.list_for_run(session, run_id)
    return [
        EventResponse(
            id=r.id,
            run_id=r.run_id,
            event_type=r.event_type,
            payload=r.payload,
            created_at=r.created_at,
        )
        for r in rows
    ]


@router.get("/{run_id}/details", response_model=RunDetailsResponse)
def get_run_details(
    run_id: str,
    session: Session = Depends(get_db_session),
) -> RunDetailsResponse:
    run_row = _run_repo.get(session, run_id)
    if run_row is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    agent_rows = _agent_repo.list_for_run(session, run_id)
    tool_rows = _tool_repo.list_for_run(session, run_id)
    event_rows = _event_repo.list_for_run(session, run_id)

    return RunDetailsResponse(
        run_id=run_row.run_id,
        workflow_id=run_row.workflow_id,
        status=run_row.status,
        input=run_row.input,
        output=run_row.output,
        error=run_row.error,
        created_at=run_row.created_at,
        updated_at=run_row.updated_at,
        agent_results=[
            AgentResultResponse(
                id=r.id,
                run_id=r.run_id,
                agent_id=r.agent_id,
                output=r.output,
                created_at=r.created_at,
            )
            for r in agent_rows
        ],
        tool_calls=[
            ToolCallResponse(
                id=r.id,
                run_id=r.run_id,
                tool_name=r.tool_name,
                input=r.input,
                output=r.output,
                is_error=r.is_error,
                created_at=r.created_at,
            )
            for r in tool_rows
        ],
        events=[
            EventResponse(
                id=r.id,
                run_id=r.run_id,
                event_type=r.event_type,
                payload=r.payload,
                created_at=r.created_at,
            )
            for r in event_rows
        ],
    )

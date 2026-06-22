"""Workflow listing and detail endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_workflow_registry
from api.schemas.workflow import WorkflowResponse
from platform.registries.workflow_registry import WorkflowRegistry

router = APIRouter()


@router.get("/", response_model=list[WorkflowResponse])
async def list_workflows(
    wf_registry: WorkflowRegistry = Depends(get_workflow_registry),
) -> list[WorkflowResponse]:
    return [
        WorkflowResponse(
            workflow_id=wf.workflow_id,
            name=wf.name,
            description=wf.description,
            pattern=wf.pattern.value,
            hitl_enabled=wf.hitl_enabled,
        )
        for wf in wf_registry.list_all()
    ]


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: str,
    wf_registry: WorkflowRegistry = Depends(get_workflow_registry),
) -> WorkflowResponse:
    wf = wf_registry.get(workflow_id)
    return WorkflowResponse(
        workflow_id=wf.workflow_id,
        name=wf.name,
        description=wf.description,
        pattern=wf.pattern.value,
        hitl_enabled=wf.hitl_enabled,
    )

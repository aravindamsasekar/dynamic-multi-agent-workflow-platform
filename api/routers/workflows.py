"""Workflow listing and detail endpoints."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def list_workflows():
    # TODO: inject WorkflowRegistry, return list[WorkflowResponse]
    raise NotImplementedError


@router.get("/{workflow_id}")
async def get_workflow(workflow_id: str):
    # TODO: inject WorkflowRegistry, return WorkflowResponse or 404
    raise NotImplementedError

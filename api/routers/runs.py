"""Workflow run trigger and status endpoints."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.post("/")
async def create_run():
    # TODO: accept RunRequest, call orchestrator.run(workflow_id, input), return RunResponse
    raise NotImplementedError


@router.get("/{run_id}")
async def get_run(run_id: str):
    # TODO: inject RunManager, return RunStatusResponse or 404
    raise NotImplementedError

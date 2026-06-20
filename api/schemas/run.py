"""Run API schemas."""

from __future__ import annotations

from pydantic import BaseModel


class RunRequest(BaseModel):
    workflow_id: str
    input: str


class RunResponse(BaseModel):
    run_id: str
    workflow_id: str
    status: str


class RunStatusResponse(BaseModel):
    run_id: str
    workflow_id: str
    status: str
    output: str | None = None
    error: str | None = None

"""Run API schemas."""

from __future__ import annotations

from typing import Any, Union

from pydantic import BaseModel


class RunRequest(BaseModel):
    workflow_id: str
    # Accept either a plain string (existing workflows) or a structured dict
    # (structured workflows such as pr_review).  The orchestrator normalises
    # dicts to a JSON string before handing off to pattern executors.
    input: Union[str, dict[str, Any]]


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


class ApprovalRequest(BaseModel):
    comment: str = ""


class RejectionRequest(BaseModel):
    reason: str = ""

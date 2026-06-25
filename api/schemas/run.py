"""Run API schemas."""

from __future__ import annotations

from datetime import datetime
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


class RunListItemResponse(BaseModel):
    run_id: str
    workflow_id: str
    status: str
    created_at: datetime
    updated_at: datetime


class AgentResultResponse(BaseModel):
    id: int
    run_id: str
    agent_id: str
    output: str
    created_at: datetime


class ToolCallResponse(BaseModel):
    id: int
    run_id: str
    tool_name: str
    input: str | None
    output: str | None
    is_error: bool
    created_at: datetime


class EventResponse(BaseModel):
    id: int
    run_id: str
    event_type: str
    payload: str | None
    created_at: datetime


class RunDetailsResponse(BaseModel):
    run_id: str
    workflow_id: str
    status: str
    input: str
    output: str | None
    error: str | None
    created_at: datetime
    updated_at: datetime
    agent_results: list[AgentResultResponse]
    tool_calls: list[ToolCallResponse]
    events: list[EventResponse]

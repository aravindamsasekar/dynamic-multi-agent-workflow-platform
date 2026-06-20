"""Workflow API schemas."""

from __future__ import annotations

from pydantic import BaseModel


class WorkflowResponse(BaseModel):
    workflow_id: str
    name: str
    description: str
    pattern: str
    hitl_enabled: bool

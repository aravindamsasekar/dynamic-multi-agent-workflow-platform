"""Workflow-related models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from platform.core.models.agent import AgentResult


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"


class PatternType(str, Enum):
    PARALLEL_SPECIALIST = "parallel_specialist"
    ROUTER = "router"
    PLANNER_EXECUTOR_OBSERVER = "planner_executor_observer"


class WorkflowDefinition(BaseModel):
    workflow_id: str
    name: str
    description: str = ""
    pattern: PatternType
    agent_ids: list[str] = []
    pattern_config: dict[str, Any] = {}
    hitl_enabled: bool = False
    policy_config: dict[str, Any] = {}


class WorkflowRun(BaseModel):
    run_id: str
    workflow_id: str
    status: RunStatus = RunStatus.PENDING
    input: str
    output: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class WorkflowResult(BaseModel):
    run_id: str
    workflow_id: str
    output: str
    agent_results: list[AgentResult] = []
    status: RunStatus = RunStatus.COMPLETED

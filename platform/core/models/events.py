"""Workflow event models for observability."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    WORKFLOW_STARTED = "workflow_started"
    AGENT_CALLED = "agent_called"
    AGENT_COMPLETED = "agent_completed"
    TOOL_CALLED = "tool_called"
    TOOL_COMPLETED = "tool_completed"
    POLICY_VIOLATION = "policy_violation"
    HITL_REQUESTED = "hitl_requested"
    HITL_RESOLVED = "hitl_resolved"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"


class WorkflowEvent(BaseModel):
    event_type: EventType
    run_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: dict[str, Any] = {}


class WorkflowStartedEvent(WorkflowEvent):
    event_type: EventType = EventType.WORKFLOW_STARTED


class AgentCalledEvent(WorkflowEvent):
    event_type: EventType = EventType.AGENT_CALLED


class AgentCompletedEvent(WorkflowEvent):
    event_type: EventType = EventType.AGENT_COMPLETED


class ToolCalledEvent(WorkflowEvent):
    event_type: EventType = EventType.TOOL_CALLED


class ToolCompletedEvent(WorkflowEvent):
    event_type: EventType = EventType.TOOL_COMPLETED


class PolicyViolationEvent(WorkflowEvent):
    event_type: EventType = EventType.POLICY_VIOLATION


class HITLRequestedEvent(WorkflowEvent):
    event_type: EventType = EventType.HITL_REQUESTED


class HITLResolvedEvent(WorkflowEvent):
    event_type: EventType = EventType.HITL_RESOLVED


class WorkflowCompletedEvent(WorkflowEvent):
    event_type: EventType = EventType.WORKFLOW_COMPLETED


class WorkflowFailedEvent(WorkflowEvent):
    event_type: EventType = EventType.WORKFLOW_FAILED

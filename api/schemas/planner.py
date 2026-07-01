"""Pydantic request/response schemas for the planner API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Union

from pydantic import BaseModel


class GeneratePlanRequest(BaseModel):
    goal: str


class GoalAnalysisResponse(BaseModel):
    required_capabilities: list[str]
    risk_level: str
    confidence: float
    reasoning: str
    constraints: list[str]
    requires_hitl: bool


class GuardrailConfigResponse(BaseModel):
    rule_type: str
    config: dict[str, Any]
    reason: str


class ValidationErrorResponse(BaseModel):
    code: str
    message: str


class ValidationWarningResponse(BaseModel):
    code: str
    message: str


class ValidationResultResponse(BaseModel):
    is_valid: bool
    errors: list[ValidationErrorResponse]
    warnings: list[ValidationWarningResponse]


class RuntimeAgentResponse(BaseModel):
    id: str
    name: str
    description: str
    capabilities: list[str]
    tool_names: list[str]
    system_prompt: str
    generated: bool


class PermissionSummaryResponse(BaseModel):
    id: str
    risk_level: str


class InstallSuggestionResponse(BaseModel):
    extension_id: str
    name: str
    description: str
    capabilities_provided: list[str]
    permissions: list[PermissionSummaryResponse]


class GeneratePlanResponse(BaseModel):
    plan_id: str
    goal: str
    status: str
    executable: bool
    task_label: str
    goal_analysis: GoalAnalysisResponse
    selected_pattern: str
    selected_agents: list[str]
    runtime_agents: list[RuntimeAgentResponse]
    selected_tools: list[str]
    guardrails: list[GuardrailConfigResponse]
    hitl_required: bool
    warnings: list[str]
    explanation: str
    estimated_complexity: str
    estimated_duration_seconds: int
    validation: ValidationResultResponse
    missing_capabilities: list[str] = []
    install_suggestions: list[InstallSuggestionResponse] = []
    unsupported: bool = False


class ApprovePlanRequest(BaseModel):
    input_data: Union[str, dict[str, Any]] = ""


class RejectPlanRequest(BaseModel):
    reason: str = ""


class PlanStatusResponse(BaseModel):
    plan_id: str
    goal: str
    status: str
    execution_run_id: str | None
    created_at: datetime
    updated_at: datetime


class ExecutionResponse(BaseModel):
    plan_id: str
    run_id: str
    status: str
    output: str | None = None

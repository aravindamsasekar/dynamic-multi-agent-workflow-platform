"""Planner API router — generate, preview, approve, and execute workflow plans."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import (
    get_capability_registry,
    get_db_session,
    get_execution_adapter,
    get_planner_service,
)
from api.schemas.planner import (
    ApprovePlanRequest,
    ExecutionResponse,
    GeneratePlanRequest,
    GeneratePlanResponse,
    GoalAnalysisResponse,
    GuardrailConfigResponse,
    PlanStatusResponse,
    RejectPlanRequest,
    ValidationErrorResponse,
    ValidationResultResponse,
    ValidationWarningResponse,
)
from platform.persistence.repositories.plan_repo import PlanRepository
from platform.planner.execution_adapter import ExecutionAdapter
from platform.planner.models import GeneratedWorkflowPlan, PlannerError, ValidationResult
from platform.planner.planner_service import PlannerService
from platform.planner.serialization import plan_from_json, validation_from_json

router = APIRouter()
_plan_repo = PlanRepository()


def _to_plan_response(
    plan: GeneratedWorkflowPlan,
    validation: ValidationResult,
    status: str,
) -> GeneratePlanResponse:
    a = plan.goal_analysis
    return GeneratePlanResponse(
        plan_id=plan.plan_id,
        goal=plan.user_goal,
        status=status,
        task_label=plan.task_label,
        goal_analysis=GoalAnalysisResponse(
            required_capabilities=a.required_capabilities,
            risk_level=a.risk_level.value,
            confidence=a.confidence,
            reasoning=a.reasoning,
            constraints=a.constraints,
            requires_hitl=a.requires_hitl,
        ),
        selected_pattern=plan.selected_pattern,
        selected_agents=plan.selected_agents,
        selected_tools=plan.selected_tools,
        guardrails=[
            GuardrailConfigResponse(rule_type=g.rule_type, config=g.config, reason=g.reason)
            for g in plan.guardrails
        ],
        hitl_required=plan.hitl_required,
        warnings=plan.warnings,
        explanation=plan.explanation,
        estimated_complexity=plan.estimated_complexity,
        estimated_duration_seconds=plan.estimated_duration_seconds,
        validation=ValidationResultResponse(
            is_valid=validation.is_valid,
            errors=[ValidationErrorResponse(code=e.code, message=e.message) for e in validation.errors],
            warnings=[ValidationWarningResponse(code=w.code, message=w.message) for w in validation.warnings],
        ),
    )


@router.post("/generate", response_model=GeneratePlanResponse, status_code=201)
async def generate_plan(
    request: GeneratePlanRequest,
    planner: PlannerService = Depends(get_planner_service),
    session: Session = Depends(get_db_session),
) -> GeneratePlanResponse:
    """Analyze the goal, build a plan, persist it, and return the plan with validation."""
    try:
        plan, validation = await planner.generate(request.goal)
    except PlannerError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    _plan_repo.create(session, plan, validation)
    session.commit()
    return _to_plan_response(plan, validation, status="pending_review")


@router.get("/{plan_id}", response_model=GeneratePlanResponse)
async def get_plan(
    plan_id: str,
    session: Session = Depends(get_db_session),
) -> GeneratePlanResponse:
    """Retrieve a previously generated plan by ID."""
    row = _plan_repo.get(session, plan_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Plan '{plan_id}' not found")
    plan = plan_from_json(row.plan_json)
    validation = validation_from_json(row.validation_json)
    return _to_plan_response(plan, validation, status=row.status)


@router.post("/{plan_id}/approve", response_model=ExecutionResponse)
async def approve_plan(
    plan_id: str,
    request: ApprovePlanRequest,
    adapter: ExecutionAdapter = Depends(get_execution_adapter),
    session: Session = Depends(get_db_session),
) -> ExecutionResponse:
    """Approve a pending plan and execute it via the V2 orchestrator."""
    row = _plan_repo.get(session, plan_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Plan '{plan_id}' not found")
    if row.status != "pending_review":
        raise HTTPException(
            status_code=409,
            detail=f"Plan cannot be approved (current status: {row.status})",
        )
    plan = plan_from_json(row.plan_json)
    result = await adapter.execute(plan, request.input_data)
    _plan_repo.update_status(session, plan_id, "executed", execution_run_id=result.run_id)
    session.commit()
    return ExecutionResponse(
        plan_id=plan_id,
        run_id=result.run_id,
        status=result.status.value,
        output=result.output,
    )


@router.post("/{plan_id}/reject", response_model=PlanStatusResponse)
async def reject_plan(
    plan_id: str,
    request: RejectPlanRequest,
    session: Session = Depends(get_db_session),
) -> PlanStatusResponse:
    """Reject a pending plan so it will not be executed."""
    row = _plan_repo.get(session, plan_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Plan '{plan_id}' not found")
    if row.status != "pending_review":
        raise HTTPException(
            status_code=409,
            detail=f"Plan cannot be rejected (current status: {row.status})",
        )
    _plan_repo.update_status(session, plan_id, "rejected")
    session.commit()
    row = _plan_repo.get(session, plan_id)
    return PlanStatusResponse(
        plan_id=row.plan_id,
        goal=row.goal,
        status=row.status,
        execution_run_id=row.execution_run_id,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )

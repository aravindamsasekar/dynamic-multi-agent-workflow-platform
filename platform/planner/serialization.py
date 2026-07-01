"""JSON serialization helpers for planner models."""

from __future__ import annotations

import json

from platform.planner.models import (
    GeneratedWorkflowPlan,
    GoalAnalysis,
    GuardrailConfig,
    RiskLevel,
    RuntimeAgentDefinition,
    ValidationError,
    ValidationResult,
    ValidationWarning,
)


def plan_to_json(plan: GeneratedWorkflowPlan) -> str:
    """Serialize GeneratedWorkflowPlan to a JSON string."""
    analysis = plan.goal_analysis
    d = {
        "plan_id": plan.plan_id,
        "user_goal": plan.user_goal,
        "goal_analysis": {
            "required_capabilities": analysis.required_capabilities,
            "missing_capabilities": analysis.missing_capabilities,
            "risk_level": analysis.risk_level.value,
            "confidence": analysis.confidence,
            "reasoning": analysis.reasoning,
            "constraints": analysis.constraints,
            "requires_hitl": analysis.requires_hitl,
        },
        "selected_pattern": plan.selected_pattern,
        "selected_agents": plan.selected_agents,
        "runtime_agents": [a.to_dict() for a in plan.runtime_agents],
        "selected_tools": plan.selected_tools,
        "guardrails": [
            {"rule_type": g.rule_type, "config": g.config, "reason": g.reason}
            for g in plan.guardrails
        ],
        "hitl_required": plan.hitl_required,
        "warnings": plan.warnings,
        "explanation": plan.explanation,
        "estimated_complexity": plan.estimated_complexity,
        "estimated_duration_seconds": plan.estimated_duration_seconds,
        "task_label": plan.task_label,
    }
    return json.dumps(d)


def plan_from_json(json_str: str) -> GeneratedWorkflowPlan:
    """Deserialize GeneratedWorkflowPlan from a JSON string.

    Backward compatible:
    - Old rows without 'runtime_agents': reconstruct minimal static references
      from 'selected_agents' so the plan remains coherent (all generated=False).
    - Old rows without 'task_label' or 'missing_capabilities': safe defaults.
    """
    d = json.loads(json_str)
    a = d["goal_analysis"]
    analysis = GoalAnalysis(
        required_capabilities=a["required_capabilities"],
        risk_level=RiskLevel(a["risk_level"]),
        confidence=a["confidence"],
        reasoning=a["reasoning"],
        constraints=a["constraints"],
        requires_hitl=a["requires_hitl"],
        missing_capabilities=a.get("missing_capabilities", []),
    )
    guardrails = [
        GuardrailConfig(
            rule_type=g["rule_type"],
            config=g["config"],
            reason=g["reason"],
        )
        for g in d["guardrails"]
    ]

    raw_agents = d.get("runtime_agents")
    if raw_agents is not None:
        runtime_agents = [RuntimeAgentDefinition.from_dict(a) for a in raw_agents]
    else:
        # Old rows: reconstruct static references from selected_agents.
        # Capabilities are unknown but IDs are correct for the execution path.
        runtime_agents = [
            RuntimeAgentDefinition(
                id=agent_id,
                name=agent_id,
                description="",
                capabilities=[],
                tool_names=[],
                system_prompt="",
                generated=False,
            )
            for agent_id in d.get("selected_agents", [])
        ]

    return GeneratedWorkflowPlan(
        plan_id=d["plan_id"],
        user_goal=d["user_goal"],
        goal_analysis=analysis,
        selected_pattern=d["selected_pattern"],
        selected_agents=d["selected_agents"],
        runtime_agents=runtime_agents,
        selected_tools=d["selected_tools"],
        guardrails=guardrails,
        hitl_required=d["hitl_required"],
        warnings=d["warnings"],
        explanation=d["explanation"],
        estimated_complexity=d["estimated_complexity"],
        estimated_duration_seconds=d["estimated_duration_seconds"],
        task_label=d.get("task_label", ""),
    )


def validation_to_json(result: ValidationResult) -> str:
    """Serialize ValidationResult to a JSON string."""
    d = {
        "is_valid": result.is_valid,
        "errors": [{"code": e.code, "message": e.message} for e in result.errors],
        "warnings": [{"code": w.code, "message": w.message} for w in result.warnings],
    }
    return json.dumps(d)


def validation_from_json(json_str: str) -> ValidationResult:
    """Deserialize ValidationResult from a JSON string."""
    d = json.loads(json_str)
    return ValidationResult(
        is_valid=d["is_valid"],
        errors=[ValidationError(code=e["code"], message=e["message"]) for e in d["errors"]],
        warnings=[ValidationWarning(code=w["code"], message=w["message"]) for w in d["warnings"]],
    )

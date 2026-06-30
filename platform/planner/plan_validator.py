"""PlanValidator — deterministic validation of a GeneratedWorkflowPlan."""

from __future__ import annotations

from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.models import (
    GeneratedWorkflowPlan,
    OperationType,
    RiskLevel,
    TaskType,
    ValidationError,
    ValidationResult,
    ValidationWarning,
)

_HIGH_RISK_LEVELS = (RiskLevel.HIGH, RiskLevel.CRITICAL)

# Confidence thresholds
_WARN_CONFIDENCE = 0.70
_ERROR_CONFIDENCE = 0.30


class PlanValidator:
    """Validates a GeneratedWorkflowPlan against the live CapabilityRegistry.

    No LLM calls. All checks are deterministic. Returns a ValidationResult
    with errors (blocking) and warnings (non-blocking).
    """

    def validate(
        self, plan: GeneratedWorkflowPlan, registry: CapabilityRegistry
    ) -> ValidationResult:
        errors: list[ValidationError] = []
        warnings: list[ValidationWarning] = []

        # ------------------------------------------------------------------
        # 1. Task type
        # ------------------------------------------------------------------
        if plan.goal_analysis.task_type == TaskType.UNSUPPORTED:
            errors.append(ValidationError(
                code="UNSUPPORTED_TASK_TYPE",
                message=(
                    f"Goal type '{plan.goal_analysis.task_type.value}' is not supported "
                    "in V3.1. Only 'code_review' is supported."
                ),
            ))
            # Short-circuit: remaining checks are meaningless for unsupported goals.
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # ------------------------------------------------------------------
        # 2. Confidence
        # ------------------------------------------------------------------
        conf = plan.goal_analysis.confidence
        if conf < _ERROR_CONFIDENCE:
            errors.append(ValidationError(
                code="CONFIDENCE_TOO_LOW",
                message=(
                    f"Plan confidence {conf:.2f} is below the minimum threshold "
                    f"({_ERROR_CONFIDENCE:.2f}). Clarify the goal before retrying."
                ),
            ))
        elif conf < _WARN_CONFIDENCE:
            warnings.append(ValidationWarning(
                code="LOW_CONFIDENCE",
                message=(
                    f"Plan confidence {conf:.2f} is below {_WARN_CONFIDENCE:.2f}. "
                    "Review the goal for clarity."
                ),
            ))

        # ------------------------------------------------------------------
        # 3. Pattern
        # ------------------------------------------------------------------
        if not plan.selected_pattern:
            errors.append(ValidationError(
                code="NO_PATTERN_SELECTED",
                message="No execution pattern was selected for this plan.",
            ))
        elif registry.get_pattern(plan.selected_pattern) is None:
            errors.append(ValidationError(
                code="MISSING_PATTERN",
                message=(
                    f"Selected pattern {plan.selected_pattern!r} is not registered "
                    "in the capability registry."
                ),
            ))

        # ------------------------------------------------------------------
        # 4. Agents
        # ------------------------------------------------------------------
        if not plan.selected_agents:
            errors.append(ValidationError(
                code="NO_AGENTS_SELECTED",
                message="No agents were selected for this plan.",
            ))
        else:
            for agent_id in plan.selected_agents:
                if registry.get_agent(agent_id) is None:
                    errors.append(ValidationError(
                        code="MISSING_AGENT",
                        message=(
                            f"Selected agent {agent_id!r} is not registered "
                            "in the capability registry."
                        ),
                    ))

        # ------------------------------------------------------------------
        # 5. Tools
        # ------------------------------------------------------------------
        if not plan.selected_tools:
            errors.append(ValidationError(
                code="NO_TOOLS_SELECTED",
                message="No tools were selected for this plan.",
            ))
        else:
            for tool_name in plan.selected_tools:
                tool_desc = registry.get_tool(tool_name)
                if tool_desc is None:
                    errors.append(ValidationError(
                        code="MISSING_TOOL",
                        message=(
                            f"Selected tool {tool_name!r} is not registered "
                            "in the capability registry."
                        ),
                    ))
                elif (
                    tool_desc.operation_type == OperationType.WRITE
                    and not plan.hitl_required
                ):
                    warnings.append(ValidationWarning(
                        code="WRITE_TOOL_WITHOUT_HITL",
                        message=(
                            f"Tool {tool_name!r} is a WRITE operation "
                            "but HITL is not enabled for this plan."
                        ),
                    ))

        # ------------------------------------------------------------------
        # 6. Capability coverage
        # ------------------------------------------------------------------
        selected_set = set(plan.selected_agents)
        for cap in plan.goal_analysis.required_capabilities:
            covering = registry.find_agents_by_capability(cap)
            if not any(a.agent_id in selected_set for a in covering):
                warnings.append(ValidationWarning(
                    code="CAPABILITY_UNMATCHED",
                    message=(
                        f"No selected agent covers required capability {cap!r}."
                    ),
                ))

        # ------------------------------------------------------------------
        # 7. HITL recommendations
        # ------------------------------------------------------------------
        if plan.goal_analysis.risk_level in _HIGH_RISK_LEVELS and not plan.hitl_required:
            warnings.append(ValidationWarning(
                code="HITL_RECOMMENDED",
                message=(
                    f"Risk level is '{plan.goal_analysis.risk_level.value}'. "
                    "Consider enabling HITL before approving this plan."
                ),
            ))

        if plan.goal_analysis.confidence < 0.5 and not plan.hitl_required:
            warnings.append(ValidationWarning(
                code="HITL_RECOMMENDED_LOW_CONFIDENCE",
                message=(
                    f"Low confidence ({plan.goal_analysis.confidence:.2f}). "
                    "Consider enabling HITL for human verification."
                ),
            ))

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

"""PlanValidator — deterministic validation of a GeneratedWorkflowPlan."""

from __future__ import annotations

from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.models import (
    GeneratedWorkflowPlan,
    OperationType,
    RiskLevel,
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

    Backward compatibility: plans without runtime_agents (old rows with
    runtime_agents=[]) fall back to selected_agents for agent and dataflow
    checks. Plans with runtime_agents use the full team for all checks.
    """

    def validate(
        self, plan: GeneratedWorkflowPlan, registry: CapabilityRegistry
    ) -> ValidationResult:
        errors: list[ValidationError] = []
        warnings: list[ValidationWarning] = []

        # ------------------------------------------------------------------
        # 1. Required capabilities — gate everything else
        # ------------------------------------------------------------------
        if not plan.goal_analysis.required_capabilities:
            errors.append(ValidationError(
                code="NO_REQUIRED_CAPABILITIES",
                message=(
                    "No required capabilities were identified for this goal. "
                    "The goal may be unsupported or the planner did not recognise "
                    "any applicable capabilities. Clarify the goal and retry."
                ),
            ))
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # ------------------------------------------------------------------
        # 2. Missing capabilities (hallucination filter output)
        # ------------------------------------------------------------------
        if plan.goal_analysis.missing_capabilities:
            missing = ", ".join(
                repr(c) for c in plan.goal_analysis.missing_capabilities
            )
            errors.append(ValidationError(
                code="MISSING_CAPABILITIES",
                message=(
                    f"The following capabilities were requested but are not registered "
                    f"in the capability registry: {missing}. "
                    "Remove or replace them and regenerate the plan."
                ),
            ))

        # ------------------------------------------------------------------
        # 3. Confidence
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
        # 4. Pattern
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
        # 5. Agents
        #
        # Plans with runtime_agents (Phase B+): both lists must be empty to
        # trigger NO_AGENTS_SELECTED. Only static agents are checked against
        # the registry — generated agents are not registered by design.
        #
        # Plans without runtime_agents (old rows): fall back to selected_agents.
        # ------------------------------------------------------------------
        if not plan.selected_agents and not plan.runtime_agents:
            errors.append(ValidationError(
                code="NO_AGENTS_SELECTED",
                message="No agents were selected for this plan.",
            ))
        else:
            if plan.runtime_agents:
                for agent in plan.runtime_agents:
                    if agent.generated:
                        continue  # not registered by design
                    if registry.get_agent(agent.id) is None:
                        errors.append(ValidationError(
                            code="MISSING_AGENT",
                            message=(
                                f"Selected agent {agent.id!r} is not registered "
                                "in the capability registry."
                            ),
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
        # 5b. Generated agent correctness checks
        #
        # A generated agent with no tools cannot call external services —
        # warn so the developer knows to register tools for that capability.
        # ------------------------------------------------------------------
        for agent in plan.runtime_agents:
            if not agent.generated:
                continue
            if not agent.tool_names:
                cap = agent.capabilities[0] if agent.capabilities else "?"
                warnings.append(ValidationWarning(
                    code="GENERATED_AGENT_NO_TOOLS",
                    message=(
                        f"Generated agent {agent.id!r} has no tools assigned "
                        f"for capability {cap!r}. "
                        "Register tools that expose this capability to enable full agent function."
                    ),
                ))

        # ------------------------------------------------------------------
        # 6. Tools
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
        # 6b. Generated agents with write tools require HITL
        #
        # This is a correctness/safety check, not an execution readiness check.
        # Generated agents with write-side-effect tools represent unreviewed
        # automation; HITL must be required.
        # ------------------------------------------------------------------
        for agent in plan.runtime_agents:
            if not agent.generated:
                continue
            for tool_name in agent.tool_names:
                tool_desc = registry.get_tool(tool_name)
                if (
                    tool_desc is not None
                    and tool_desc.operation_type == OperationType.WRITE
                    and not plan.hitl_required
                ):
                    errors.append(ValidationError(
                        code="GENERATED_AGENT_WRITE_WITHOUT_HITL",
                        message=(
                            f"Generated agent {agent.id!r} has write tool {tool_name!r} "
                            "but HITL is not required for this plan. "
                            "Enable HITL or remove the write tool."
                        ),
                    ))

        # ------------------------------------------------------------------
        # 7. Dataflow — agent contract satisfaction
        #
        # For each selected agent, every token it consumes that is produced by
        # *some* registered agent must also be produced by a *selected* agent.
        # Tokens not produced by any registered agent are user-provided inputs
        # and are skipped.
        #
        # Only static agents have consumes/produces contracts. Generated agents
        # have no registered contracts and are excluded from this check.
        #
        # Plans with runtime_agents: use static agents from that list.
        # Plans without runtime_agents (old rows): use selected_agents directly.
        # ------------------------------------------------------------------
        if plan.runtime_agents:
            dataflow_agents = [r.id for r in plan.runtime_agents if not r.generated]
        else:
            dataflow_agents = list(plan.selected_agents)

        all_produced = registry.all_produced_tokens()
        selected_produces: set[str] = set()
        for agent_id in dataflow_agents:
            desc = registry.get_agent(agent_id)
            if desc is not None:
                selected_produces.update(desc.produces)

        unsatisfied: set[str] = set()
        for agent_id in dataflow_agents:
            desc = registry.get_agent(agent_id)
            if desc is None:
                continue
            for token in desc.consumes:
                if token in all_produced and token not in selected_produces:
                    unsatisfied.add(token)

        for token in sorted(unsatisfied):
            errors.append(ValidationError(
                code="DATAFLOW_UNSATISFIED",
                message=(
                    f"Token {token!r} is required by the dataflow but is not "
                    "produced by any selected agent."
                ),
            ))

        # ------------------------------------------------------------------
        # 8. Capability coverage
        #
        # Plans with runtime_agents: check coverage via agent.capabilities
        # (both static and generated agents can cover capabilities).
        # Plans without runtime_agents (old rows): use registry-based lookup
        # against selected_agents.
        # ------------------------------------------------------------------
        if plan.runtime_agents:
            covered_caps = {cap for agent in plan.runtime_agents for cap in agent.capabilities}
            for cap in plan.goal_analysis.required_capabilities:
                if cap not in covered_caps:
                    warnings.append(ValidationWarning(
                        code="CAPABILITY_UNMATCHED",
                        message=(
                            f"No agent covers required capability {cap!r}."
                        ),
                    ))
        else:
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
        # 9. HITL recommendations
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

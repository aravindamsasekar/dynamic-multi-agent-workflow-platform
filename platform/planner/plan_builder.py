"""PlanBuilder — orchestrates all deterministic selectors into a GeneratedWorkflowPlan."""

from __future__ import annotations

from uuid import uuid4

from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.models import (
    GeneratedWorkflowPlan,
    GoalAnalysis,
    GuardrailConfig,
    RiskLevel,
    RuntimeAgentDefinition,
)
from platform.planner.pattern_selector import PatternSelector
from platform.planner.runtime_agent_generator import RuntimeAgentGenerator
from platform.planner.tool_selector import ToolSelector

# ---------------------------------------------------------------------------
# Internal helpers — no LLM, no I/O
# ---------------------------------------------------------------------------

_HIGH_RISK_LEVELS = (RiskLevel.HIGH, RiskLevel.CRITICAL)


def _generate_guardrails(analysis: GoalAnalysis) -> list[GuardrailConfig]:
    guardrails: list[GuardrailConfig] = []

    # Base content filter applied to all plans
    guardrails.append(GuardrailConfig(
        rule_type="content_filter",
        config={"blocked_terms": ["rm -rf", "DROP TABLE", "truncate", "delete all"]},
        reason="Block destructive commands in agent output",
    ))

    if analysis.risk_level in _HIGH_RISK_LEVELS:
        guardrails.append(GuardrailConfig(
            rule_type="content_filter",
            config={"blocked_terms": ["irreversible", "production reset", "wipe"]},
            reason=f"Additional filter for {analysis.risk_level.value} risk level",
        ))

    if "read_only" in analysis.constraints or "no_external_writes" in analysis.constraints:
        guardrails.append(GuardrailConfig(
            rule_type="tool_permission",
            config={"blocked_operations": ["write", "execute"]},
            reason="Enforce read-only constraint",
        ))

    return guardrails


def _generate_warnings(
    analysis: GoalAnalysis,
    runtime_agents: list[RuntimeAgentDefinition],
    selected_tools: list[str],
) -> list[str]:
    notes: list[str] = []

    if analysis.confidence < 0.7:
        notes.append(
            f"Low confidence ({analysis.confidence:.2f}). "
            "Consider clarifying the goal before approving."
        )

    if analysis.requires_hitl:
        notes.append(
            f"HITL enabled due to {analysis.risk_level.value} risk level."
        )

    if not runtime_agents:
        notes.append("No agents could be selected for this goal type.")

    if not selected_tools:
        notes.append("No tools could be selected for the chosen agents.")

    return notes


def _estimate_complexity(agent_count: int) -> str:
    if agent_count <= 2:
        return "low"
    if agent_count <= 4:
        return "medium"
    return "high"


def _estimate_duration_seconds(agent_count: int, tool_count: int) -> int:
    return 15 + (agent_count * 10) + (tool_count * 3)


def _generate_explanation(
    user_goal: str,
    analysis: GoalAnalysis,
    selected_pattern: str,
    runtime_agents: list[RuntimeAgentDefinition],
) -> str:
    cap_summary = ", ".join(analysis.required_capabilities[:3]) or "none"
    agents_summary = ", ".join(r.id for r in runtime_agents) or "none"
    pattern_label = selected_pattern or "none"
    return (
        f"Capability-based plan requiring {len(analysis.required_capabilities)} capabilities, "
        f"{analysis.confidence:.0%} confidence. "
        f"Selected pattern: {pattern_label}. "
        f"Agents: {agents_summary}. "
        f"Key capabilities: {cap_summary}."
    )


# ---------------------------------------------------------------------------
# PlanBuilder
# ---------------------------------------------------------------------------


class PlanBuilder:
    """Orchestrates agent, tool, and pattern selection into a GeneratedWorkflowPlan.

    No LLM calls. Accepts a CapabilityRegistry and produces a plan that can
    be validated (PlanValidator) and later executed (Phase E).

    Invariant maintained by this class:
        plan.selected_agents == [r.id for r in plan.runtime_agents if not r.generated]

    PlanBuilder is the sole component that constructs both runtime_agents and
    selected_agents. No other component mutates either field after build().
    """

    def __init__(self, registry: CapabilityRegistry) -> None:
        self._registry = registry
        self._runtime_agent_generator = RuntimeAgentGenerator(registry)
        self._tool_selector = ToolSelector()
        self._pattern_selector = PatternSelector()

    def build(self, user_goal: str, analysis: GoalAnalysis) -> GeneratedWorkflowPlan:
        """Build a GeneratedWorkflowPlan from a user goal and GoalAnalysis.

        All selections are deterministic — no I/O, no LLM.
        Agents are generated first; pattern and tool selection consult the full team.
        """
        # Generate plan_id first so generated agent IDs are scoped to this plan.
        # Invariant: selected_agents == [r.id for r in runtime_agents if not r.generated]
        plan_id = str(uuid4())
        runtime_agents = self._runtime_agent_generator.generate(
            analysis.required_capabilities, plan_id=plan_id
        )
        selected_agents = [r.id for r in runtime_agents if not r.generated]

        selected_pattern = self._pattern_selector.select(analysis, runtime_agents, self._registry)
        selected_tools = self._tool_selector.select(analysis, runtime_agents, self._registry)

        guardrails = _generate_guardrails(analysis)
        warnings = _generate_warnings(analysis, runtime_agents, selected_tools)
        explanation = _generate_explanation(
            user_goal, analysis, selected_pattern, runtime_agents
        )

        return GeneratedWorkflowPlan(
            plan_id=plan_id,
            user_goal=user_goal,
            goal_analysis=analysis,
            selected_pattern=selected_pattern,
            selected_agents=selected_agents,
            runtime_agents=runtime_agents,
            selected_tools=selected_tools,
            guardrails=guardrails,
            hitl_required=analysis.requires_hitl,
            warnings=warnings,
            explanation=explanation,
            estimated_complexity=_estimate_complexity(len(runtime_agents)),
            estimated_duration_seconds=_estimate_duration_seconds(
                len(runtime_agents), len(selected_tools)
            ),
        )

"""Unit tests for PlanBuilder — deterministic, no LLM."""

from __future__ import annotations

import pytest

from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.models import (
    GeneratedWorkflowPlan,
    GoalAnalysis,
    RiskLevel,
)
from platform.planner.plan_builder import PlanBuilder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> CapabilityRegistry:
    return CapabilityRegistry.build_pr_review_registry()


@pytest.fixture
def builder(registry: CapabilityRegistry) -> PlanBuilder:
    return PlanBuilder(registry=registry)


def _analysis(
    confidence: float = 0.92,
    risk_level: RiskLevel = RiskLevel.LOW,
    requires_hitl: bool = False,
    required_capabilities: list[str] | None = None,
    constraints: list[str] | None = None,
) -> GoalAnalysis:
    return GoalAnalysis(
        required_capabilities=required_capabilities or [
            "fetch_pr_data",
            "review_code_quality",
            "assess_security",
            "synthesize_findings",
        ],
        risk_level=risk_level,
        confidence=confidence,
        reasoning="Test fixture.",
        constraints=constraints if constraints is not None else ["read_only"],
        requires_hitl=requires_hitl,
    )


def _unsupported_analysis() -> GoalAnalysis:
    return GoalAnalysis(
        required_capabilities=[],
        risk_level=RiskLevel.LOW,
        confidence=0.0,
        reasoning="Goal not supported.",
        constraints=[],
        requires_hitl=False,
    )


# ---------------------------------------------------------------------------
# Basic plan shape
# ---------------------------------------------------------------------------


class TestPlanBuilderOutputShape:
    def test_returns_generated_workflow_plan(self, builder: PlanBuilder):
        plan = builder.build("Review PR #42", _analysis())
        assert isinstance(plan, GeneratedWorkflowPlan)

    def test_plan_id_is_non_empty_string(self, builder: PlanBuilder):
        plan = builder.build("Review PR #42", _analysis())
        assert isinstance(plan.plan_id, str)
        assert len(plan.plan_id) > 0

    def test_plan_ids_are_unique(self, builder: PlanBuilder):
        a = builder.build("Review PR #42", _analysis())
        b = builder.build("Review PR #42", _analysis())
        assert a.plan_id != b.plan_id

    def test_user_goal_preserved(self, builder: PlanBuilder):
        goal = "Review PR #99 for security"
        plan = builder.build(goal, _analysis())
        assert plan.user_goal == goal

    def test_goal_analysis_preserved(self, builder: PlanBuilder):
        analysis = _analysis(confidence=0.88)
        plan = builder.build("Review PR #42", analysis)
        assert plan.goal_analysis is analysis

    def test_has_non_empty_explanation(self, builder: PlanBuilder):
        plan = builder.build("Review PR #42", _analysis())
        assert isinstance(plan.explanation, str)
        assert len(plan.explanation) > 0

    def test_explanation_mentions_capabilities(self, builder: PlanBuilder):
        plan = builder.build("Review PR #42", _analysis())
        assert "capabilities" in plan.explanation

    def test_explanation_mentions_pattern(self, builder: PlanBuilder):
        plan = builder.build("Review PR #42", _analysis())
        assert "parallel_specialist" in plan.explanation

    def test_explanation_mentions_agents(self, builder: PlanBuilder):
        plan = builder.build("Review PR #42", _analysis())
        assert "pr_data_agent" in plan.explanation


# ---------------------------------------------------------------------------
# No LLM calls
# ---------------------------------------------------------------------------


class TestPlanBuilderNoLLM:
    def test_build_completes_without_llm_provider(self, registry: CapabilityRegistry):
        # PlanBuilder does not accept ILLMProvider — this confirms no LLM dependency.
        builder = PlanBuilder(registry=registry)
        plan = builder.build("Review PR #42", _analysis())
        assert plan is not None

    def test_build_is_synchronous(self, builder: PlanBuilder):
        # build() is a plain method, not a coroutine — no async I/O.
        import inspect
        assert not inspect.iscoroutinefunction(builder.build)


# ---------------------------------------------------------------------------
# CODE_REVIEW plan — agent, tool, pattern selection
# ---------------------------------------------------------------------------


class TestPlanBuilderCodeReview:
    def test_selects_parallel_specialist_pattern(self, builder: PlanBuilder):
        plan = builder.build("Review PR #42", _analysis())
        assert plan.selected_pattern == "parallel_specialist"

    def test_selects_all_four_pr_review_agents(self, builder: PlanBuilder):
        plan = builder.build("Review PR #42", _analysis())
        expected = {"pr_data_agent", "review_specialist", "risk_specialist", "synthesis_agent"}
        assert set(plan.selected_agents) == expected

    def test_selects_exactly_four_agents(self, builder: PlanBuilder):
        plan = builder.build("Review PR #42", _analysis())
        assert len(plan.selected_agents) == 4

    def test_selects_all_five_pr_review_tools(self, builder: PlanBuilder):
        plan = builder.build("Review PR #42", _analysis())
        expected = {
            "github_get_pr",
            "github_get_files",
            "github_get_diff",
            "knowledge_search",
            "mcp_get_pr_comments",
        }
        assert set(plan.selected_tools) == expected

    def test_selects_exactly_five_tools(self, builder: PlanBuilder):
        plan = builder.build("Review PR #42", _analysis())
        assert len(plan.selected_tools) == 5

    def test_hitl_required_false_for_low_risk(self, builder: PlanBuilder):
        plan = builder.build("Review PR #42", _analysis(requires_hitl=False))
        assert plan.hitl_required is False

    def test_hitl_required_true_from_analysis(self, builder: PlanBuilder):
        plan = builder.build(
            "Review PR #42 touching production",
            _analysis(requires_hitl=True, risk_level=RiskLevel.HIGH),
        )
        assert plan.hitl_required is True


# ---------------------------------------------------------------------------
# Unsupported goal
# ---------------------------------------------------------------------------


class TestPlanBuilderUnsupported:
    def test_unsupported_goal_has_no_agents(self, builder: PlanBuilder):
        plan = builder.build("What is the weather?", _unsupported_analysis())
        assert plan.selected_agents == []

    def test_unsupported_goal_has_no_tools(self, builder: PlanBuilder):
        plan = builder.build("What is the weather?", _unsupported_analysis())
        assert plan.selected_tools == []

    def test_unsupported_goal_has_empty_pattern(self, builder: PlanBuilder):
        plan = builder.build("What is the weather?", _unsupported_analysis())
        assert plan.selected_pattern == ""

    def test_unsupported_goal_still_returns_plan(self, builder: PlanBuilder):
        plan = builder.build("What is the weather?", _unsupported_analysis())
        assert isinstance(plan, GeneratedWorkflowPlan)
        assert plan.user_goal == "What is the weather?"


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


class TestPlanBuilderGuardrails:
    def test_has_at_least_one_guardrail(self, builder: PlanBuilder):
        plan = builder.build("Review PR #42", _analysis())
        assert len(plan.guardrails) >= 1

    def test_base_content_filter_always_present(self, builder: PlanBuilder):
        plan = builder.build("Review PR #42", _analysis())
        content_filters = [g for g in plan.guardrails if g.rule_type == "content_filter"]
        assert len(content_filters) >= 1

    def test_read_only_constraint_adds_tool_permission_guardrail(self, builder: PlanBuilder):
        plan = builder.build("Review PR #42", _analysis(constraints=["read_only"]))
        tool_perm_rules = [g for g in plan.guardrails if g.rule_type == "tool_permission"]
        assert len(tool_perm_rules) >= 1
        assert "write" in tool_perm_rules[0].config.get("blocked_operations", [])

    def test_no_external_writes_adds_tool_permission_guardrail(self, builder: PlanBuilder):
        plan = builder.build("Review PR #42", _analysis(constraints=["no_external_writes"]))
        tool_perm_rules = [g for g in plan.guardrails if g.rule_type == "tool_permission"]
        assert len(tool_perm_rules) >= 1

    def test_high_risk_adds_extra_content_filter(self, builder: PlanBuilder):
        plan = builder.build(
            "Review production PR",
            _analysis(risk_level=RiskLevel.HIGH, constraints=[]),
        )
        content_filters = [g for g in plan.guardrails if g.rule_type == "content_filter"]
        assert len(content_filters) >= 2

    def test_low_risk_no_extra_content_filter(self, builder: PlanBuilder):
        plan = builder.build(
            "Review PR #42",
            _analysis(risk_level=RiskLevel.LOW, constraints=[]),
        )
        content_filters = [g for g in plan.guardrails if g.rule_type == "content_filter"]
        assert len(content_filters) == 1


# ---------------------------------------------------------------------------
# Warnings (builder-side notes)
# ---------------------------------------------------------------------------


class TestPlanBuilderWarnings:
    def test_low_confidence_generates_warning(self, builder: PlanBuilder):
        plan = builder.build("Review PR #42", _analysis(confidence=0.5))
        assert any("confidence" in w.lower() for w in plan.warnings)

    def test_high_confidence_no_confidence_warning(self, builder: PlanBuilder):
        plan = builder.build("Review PR #42", _analysis(confidence=0.95))
        assert not any("confidence" in w.lower() for w in plan.warnings)

    def test_hitl_enabled_generates_note(self, builder: PlanBuilder):
        plan = builder.build(
            "Review production PR",
            _analysis(requires_hitl=True, risk_level=RiskLevel.HIGH),
        )
        assert any("hitl" in w.lower() for w in plan.warnings)


# ---------------------------------------------------------------------------
# V3.2 new fields
# ---------------------------------------------------------------------------


class TestPlanBuilderV32Fields:
    def test_task_label_defaults_to_empty_string(self, builder: PlanBuilder):
        plan = builder.build("Review PR #42", _analysis())
        assert plan.task_label == ""

    def test_goal_analysis_missing_capabilities_defaults_to_empty(self, builder: PlanBuilder):
        plan = builder.build("Review PR #42", _analysis())
        assert plan.goal_analysis.missing_capabilities == []

    def test_goal_analysis_missing_capabilities_preserved_on_plan(self, builder: PlanBuilder):
        analysis = _analysis()
        analysis.missing_capabilities = ["unavailable_cap"]
        plan = builder.build("Review PR #42", analysis)
        assert plan.goal_analysis.missing_capabilities == ["unavailable_cap"]

    def test_agent_selection_driven_by_capabilities(self, builder: PlanBuilder):
        # A single-capability goal selects only the agent covering that capability.
        plan = builder.build("Assess security", _analysis(required_capabilities=["assess_security"]))
        assert plan.selected_agents == ["risk_specialist"]

    def test_agent_selection_preserves_encounter_order(self, builder: PlanBuilder):
        # Capability order in required_capabilities determines agent order in plan.
        plan = builder.build(
            "Review PR",
            _analysis(required_capabilities=["fetch_pr_data", "review_code_quality"]),
        )
        assert plan.selected_agents == ["pr_data_agent", "review_specialist"]

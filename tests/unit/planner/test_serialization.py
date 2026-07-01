"""Unit tests for planner serialization round-trips."""

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
from platform.planner.serialization import (
    plan_from_json,
    plan_to_json,
    validation_from_json,
    validation_to_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plan(task_label: str = "") -> GeneratedWorkflowPlan:
    return GeneratedWorkflowPlan(
        plan_id="ser-test-plan",
        user_goal="Review PR #1",
        goal_analysis=GoalAnalysis(
            required_capabilities=["fetch_pr_data", "review_code_quality"],
            risk_level=RiskLevel.LOW,
            confidence=0.88,
            reasoning="Serialization test.",
            constraints=["read_only"],
            requires_hitl=False,
        ),
        selected_pattern="parallel_specialist",
        selected_agents=["pr_data_agent", "review_specialist"],
        selected_tools=["github_get_pr", "github_get_diff"],
        guardrails=[GuardrailConfig(rule_type="content_filter", config={"k": "v"}, reason="safety")],
        hitl_required=False,
        warnings=["Low confidence."],
        explanation="A test plan.",
        estimated_complexity="low",
        estimated_duration_seconds=30,
        task_label=task_label,
    )


# ---------------------------------------------------------------------------
# Plan round-trip
# ---------------------------------------------------------------------------


class TestPlanRoundTrip:
    def test_plan_id_round_trips(self):
        plan = _make_plan()
        assert plan_from_json(plan_to_json(plan)).plan_id == plan.plan_id

    def test_user_goal_round_trips(self):
        plan = _make_plan()
        assert plan_from_json(plan_to_json(plan)).user_goal == plan.user_goal

    def test_selected_pattern_round_trips(self):
        plan = _make_plan()
        assert plan_from_json(plan_to_json(plan)).selected_pattern == plan.selected_pattern

    def test_selected_agents_round_trips(self):
        plan = _make_plan()
        assert plan_from_json(plan_to_json(plan)).selected_agents == plan.selected_agents

    def test_selected_tools_round_trips(self):
        plan = _make_plan()
        assert plan_from_json(plan_to_json(plan)).selected_tools == plan.selected_tools

    def test_guardrail_rule_type_round_trips(self):
        plan = _make_plan()
        restored = plan_from_json(plan_to_json(plan))
        assert restored.guardrails[0].rule_type == "content_filter"
        assert restored.guardrails[0].config == {"k": "v"}

    def test_warnings_round_trips(self):
        plan = _make_plan()
        assert plan_from_json(plan_to_json(plan)).warnings == plan.warnings

    def test_confidence_round_trips(self):
        plan = _make_plan()
        restored = plan_from_json(plan_to_json(plan))
        assert restored.goal_analysis.confidence == plan.goal_analysis.confidence


# ---------------------------------------------------------------------------
# task_label
# ---------------------------------------------------------------------------


class TestTaskLabelSerialization:
    def test_task_label_round_trips_non_empty(self):
        plan = _make_plan(task_label="code_review")
        restored = plan_from_json(plan_to_json(plan))
        assert restored.task_label == "code_review"

    def test_task_label_round_trips_empty_string(self):
        plan = _make_plan(task_label="")
        restored = plan_from_json(plan_to_json(plan))
        assert restored.task_label == ""

    def test_task_label_defaults_to_empty_for_old_json(self):
        """Rows serialized before V3.2 (no task_label key) deserialize cleanly."""
        plan = _make_plan()
        d = json.loads(plan_to_json(plan))
        del d["task_label"]
        restored = plan_from_json(json.dumps(d))
        assert restored.task_label == ""

    def test_task_label_custom_round_trips(self):
        plan = _make_plan(task_label="custom")
        assert plan_from_json(plan_to_json(plan)).task_label == "custom"


# ---------------------------------------------------------------------------
# missing_capabilities
# ---------------------------------------------------------------------------


class TestMissingCapabilitiesSerialization:
    def test_missing_capabilities_empty_round_trips(self):
        plan = _make_plan()
        restored = plan_from_json(plan_to_json(plan))
        assert restored.goal_analysis.missing_capabilities == []

    def test_missing_capabilities_non_empty_round_trips(self):
        plan = _make_plan()
        plan.goal_analysis.missing_capabilities = ["unknown_cap", "another"]
        restored = plan_from_json(plan_to_json(plan))
        assert restored.goal_analysis.missing_capabilities == ["unknown_cap", "another"]

    def test_missing_capabilities_defaults_to_empty_for_old_json(self):
        """Rows serialized before V3.2 (no missing_capabilities key) deserialize cleanly."""
        plan = _make_plan()
        d = json.loads(plan_to_json(plan))
        del d["goal_analysis"]["missing_capabilities"]
        restored = plan_from_json(json.dumps(d))
        assert restored.goal_analysis.missing_capabilities == []

    def test_old_json_with_task_type_key_is_ignored(self):
        """Rows written before Phase E (with task_type in goal_analysis) deserialize cleanly."""
        plan = _make_plan()
        d = json.loads(plan_to_json(plan))
        d["goal_analysis"]["task_type"] = "code_review"
        restored = plan_from_json(json.dumps(d))
        assert not hasattr(restored.goal_analysis, "task_type")
        assert restored.goal_analysis.required_capabilities == plan.goal_analysis.required_capabilities


# ---------------------------------------------------------------------------
# Validation round-trip (unchanged — regression guard)
# ---------------------------------------------------------------------------


class TestValidationRoundTrip:
    def test_is_valid_round_trips(self):
        v = ValidationResult(is_valid=True, errors=[], warnings=[])
        assert validation_from_json(validation_to_json(v)).is_valid is True

    def test_error_code_round_trips(self):
        v = ValidationResult(
            is_valid=False,
            errors=[ValidationError(code="MISSING_AGENT", message="Missing.")],
            warnings=[],
        )
        restored = validation_from_json(validation_to_json(v))
        assert restored.errors[0].code == "MISSING_AGENT"

    def test_warning_code_round_trips(self):
        v = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[ValidationWarning(code="LOW_CONFIDENCE", message="Low.")],
        )
        restored = validation_from_json(validation_to_json(v))
        assert restored.warnings[0].code == "LOW_CONFIDENCE"


# ---------------------------------------------------------------------------
# runtime_agents serialization (Phase B)
# ---------------------------------------------------------------------------


def _make_runtime_agent(agent_id: str, generated: bool = False) -> RuntimeAgentDefinition:
    return RuntimeAgentDefinition(
        id=agent_id,
        name=f"{agent_id} name",
        description=f"{agent_id} description",
        capabilities=[f"{agent_id}_cap"],
        tool_names=[] if not generated else ["some_tool"],
        system_prompt="" if not generated else f"You handle {agent_id}.",
        generated=generated,
    )


class TestRuntimeAgentsSerialization:
    def test_runtime_agents_empty_round_trips(self):
        plan = _make_plan()
        restored = plan_from_json(plan_to_json(plan))
        assert restored.runtime_agents == []

    def test_runtime_agents_non_empty_round_trips(self):
        plan = _make_plan()
        plan.runtime_agents = [_make_runtime_agent("pr_data_agent", generated=False)]
        restored = plan_from_json(plan_to_json(plan))
        assert len(restored.runtime_agents) == 1
        assert restored.runtime_agents[0].id == "pr_data_agent"

    def test_static_agent_generated_flag_round_trips(self):
        plan = _make_plan()
        plan.runtime_agents = [_make_runtime_agent("static_agent", generated=False)]
        restored = plan_from_json(plan_to_json(plan))
        assert restored.runtime_agents[0].generated is False

    def test_generated_agent_flag_round_trips(self):
        plan = _make_plan()
        plan.runtime_agents = [_make_runtime_agent("gen_agent", generated=True)]
        restored = plan_from_json(plan_to_json(plan))
        assert restored.runtime_agents[0].generated is True

    def test_generated_agent_tool_names_round_trip(self):
        plan = _make_plan()
        plan.runtime_agents = [_make_runtime_agent("gen_agent", generated=True)]
        restored = plan_from_json(plan_to_json(plan))
        assert restored.runtime_agents[0].tool_names == ["some_tool"]

    def test_old_json_without_runtime_agents_key_reconstructs_from_selected_agents(self):
        """Old rows without 'runtime_agents' key reconstruct from 'selected_agents'."""
        plan = _make_plan()
        d = json.loads(plan_to_json(plan))
        del d["runtime_agents"]
        restored = plan_from_json(json.dumps(d))
        # Should reconstruct from selected_agents = ["pr_data_agent", "review_specialist"]
        assert len(restored.runtime_agents) == 2
        ids = {r.id for r in restored.runtime_agents}
        assert "pr_data_agent" in ids
        assert "review_specialist" in ids

    def test_reconstructed_old_plan_agents_are_not_generated(self):
        plan = _make_plan()
        d = json.loads(plan_to_json(plan))
        del d["runtime_agents"]
        restored = plan_from_json(json.dumps(d))
        assert all(not r.generated for r in restored.runtime_agents)

    def test_plan_serialization_includes_both_agent_fields(self):
        plan = _make_plan()
        plan.runtime_agents = [_make_runtime_agent("pr_data_agent")]
        d = json.loads(plan_to_json(plan))
        assert "selected_agents" in d
        assert "runtime_agents" in d

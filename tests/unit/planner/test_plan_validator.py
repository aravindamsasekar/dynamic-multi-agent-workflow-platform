"""Unit tests for PlanValidator — deterministic, no LLM."""

from __future__ import annotations

import pytest

from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.models import (
    AgentCapabilityDescriptor,
    GoalAnalysis,
    GeneratedWorkflowPlan,
    GuardrailConfig,
    PatternCapabilityDescriptor,
    RiskLevel,
    ValidationResult,
)
from platform.planner.plan_builder import PlanBuilder
from platform.planner.plan_validator import PlanValidator


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> CapabilityRegistry:
    return CapabilityRegistry.build_pr_review_registry()


@pytest.fixture
def builder(registry: CapabilityRegistry) -> PlanBuilder:
    return PlanBuilder(registry=registry)


@pytest.fixture
def validator() -> PlanValidator:
    return PlanValidator()


def _analysis(
    confidence: float = 0.92,
    risk_level: RiskLevel = RiskLevel.LOW,
    requires_hitl: bool = False,
    required_capabilities: list[str] | None = None,
    constraints: list[str] | None = None,
    missing_capabilities: list[str] | None = None,
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
        missing_capabilities=missing_capabilities or [],
    )


def _make_plan(
    analysis: GoalAnalysis,
    agents: list[str],
    pattern: str = "parallel_specialist",
    tools: list[str] | None = None,
) -> GeneratedWorkflowPlan:
    """Build a minimal GeneratedWorkflowPlan for targeted validator tests."""
    return GeneratedWorkflowPlan(
        plan_id="test-plan",
        user_goal="test goal",
        goal_analysis=analysis,
        selected_pattern=pattern,
        selected_agents=agents,
        selected_tools=tools or [],
        guardrails=[],
        hitl_required=analysis.requires_hitl,
        warnings=[],
        explanation="test",
        estimated_complexity="low",
        estimated_duration_seconds=15,
    )


def _unsupported_analysis() -> GoalAnalysis:
    return GoalAnalysis(
        required_capabilities=[],
        risk_level=RiskLevel.LOW,
        confidence=0.0,
        reasoning="Not supported.",
        constraints=[],
        requires_hitl=False,
    )


def _build_valid_plan(builder: PlanBuilder) -> GeneratedWorkflowPlan:
    return builder.build("Review PR #42", _analysis())


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


class TestValidationResultShape:
    def test_returns_validation_result(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        plan = _build_valid_plan(builder)
        result = validator.validate(plan, registry)
        assert isinstance(result, ValidationResult)

    def test_result_has_is_valid(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        plan = _build_valid_plan(builder)
        result = validator.validate(plan, registry)
        assert isinstance(result.is_valid, bool)

    def test_result_has_errors_list(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        plan = _build_valid_plan(builder)
        result = validator.validate(plan, registry)
        assert isinstance(result.errors, list)

    def test_result_has_warnings_list(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        plan = _build_valid_plan(builder)
        result = validator.validate(plan, registry)
        assert isinstance(result.warnings, list)


# ---------------------------------------------------------------------------
# Happy path — valid CODE_REVIEW plan
# ---------------------------------------------------------------------------


class TestValidatorHappyPath:
    def test_valid_code_review_plan_is_valid(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        plan = _build_valid_plan(builder)
        result = validator.validate(plan, registry)
        assert result.is_valid is True

    def test_valid_plan_has_no_errors(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        plan = _build_valid_plan(builder)
        result = validator.validate(plan, registry)
        assert result.errors == []

    def test_valid_plan_with_high_confidence_has_no_confidence_warning(
        self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry
    ):
        plan = builder.build("Review PR #42", _analysis(confidence=0.95))
        result = validator.validate(plan, registry)
        codes = {w.code for w in result.warnings}
        assert "LOW_CONFIDENCE" not in codes
        assert "CONFIDENCE_TOO_LOW" not in codes


# ---------------------------------------------------------------------------
# NO_REQUIRED_CAPABILITIES (replaces UNSUPPORTED_TASK_TYPE)
# ---------------------------------------------------------------------------


class TestValidatorNoRequiredCapabilities:
    def test_empty_required_capabilities_is_invalid(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        plan = builder.build("Random goal", _unsupported_analysis())
        result = validator.validate(plan, registry)
        assert result.is_valid is False

    def test_empty_capabilities_sets_error_code(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        plan = builder.build("Random goal", _unsupported_analysis())
        result = validator.validate(plan, registry)
        codes = {e.code for e in result.errors}
        assert "NO_REQUIRED_CAPABILITIES" in codes

    def test_empty_capabilities_short_circuits(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        plan = builder.build("Random goal", _unsupported_analysis())
        result = validator.validate(plan, registry)
        # Only one error — NO_REQUIRED_CAPABILITIES — remaining checks skipped
        assert len(result.errors) == 1
        assert result.errors[0].code == "NO_REQUIRED_CAPABILITIES"

    def test_unsupported_task_type_error_code_no_longer_exists(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        plan = builder.build("Random goal", _unsupported_analysis())
        result = validator.validate(plan, registry)
        codes = {e.code for e in result.errors}
        assert "UNSUPPORTED_TASK_TYPE" not in codes

    def test_non_empty_capabilities_not_affected(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        plan = _build_valid_plan(builder)
        result = validator.validate(plan, registry)
        codes = {e.code for e in result.errors}
        assert "NO_REQUIRED_CAPABILITIES" not in codes


# ---------------------------------------------------------------------------
# Confidence validation
# ---------------------------------------------------------------------------


class TestValidatorConfidence:
    def test_confidence_below_0_7_produces_low_confidence_warning(
        self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry
    ):
        plan = builder.build("Review PR #42", _analysis(confidence=0.6))
        result = validator.validate(plan, registry)
        codes = {w.code for w in result.warnings}
        assert "LOW_CONFIDENCE" in codes

    def test_confidence_below_0_3_produces_error(
        self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry
    ):
        plan = builder.build("Review PR #42", _analysis(confidence=0.2))
        result = validator.validate(plan, registry)
        codes = {e.code for e in result.errors}
        assert "CONFIDENCE_TOO_LOW" in codes

    def test_confidence_below_0_3_plan_is_invalid(
        self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry
    ):
        plan = builder.build("Review PR #42", _analysis(confidence=0.15))
        result = validator.validate(plan, registry)
        assert result.is_valid is False

    def test_confidence_0_7_exactly_has_no_warning(
        self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry
    ):
        plan = builder.build("Review PR #42", _analysis(confidence=0.7))
        result = validator.validate(plan, registry)
        codes = {w.code for w in result.warnings}
        assert "LOW_CONFIDENCE" not in codes


# ---------------------------------------------------------------------------
# Agent validation
# ---------------------------------------------------------------------------


class TestValidatorAgents:
    def test_empty_agents_is_invalid(self, validator: PlanValidator, registry: CapabilityRegistry, builder: PlanBuilder):
        plan = _build_valid_plan(builder)
        plan.selected_agents = []
        result = validator.validate(plan, registry)
        assert result.is_valid is False
        codes = {e.code for e in result.errors}
        assert "NO_AGENTS_SELECTED" in codes

    def test_missing_agent_in_registry_is_invalid(
        self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry
    ):
        plan = _build_valid_plan(builder)
        plan.selected_agents = ["nonexistent_agent"]
        result = validator.validate(plan, registry)
        assert result.is_valid is False

    def test_missing_agent_sets_missing_agent_error_code(
        self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry
    ):
        plan = _build_valid_plan(builder)
        plan.selected_agents = ["phantom"]
        result = validator.validate(plan, registry)
        codes = {e.code for e in result.errors}
        assert "MISSING_AGENT" in codes


# ---------------------------------------------------------------------------
# Tool validation
# ---------------------------------------------------------------------------


class TestValidatorTools:
    def test_empty_tools_is_invalid(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        plan = _build_valid_plan(builder)
        plan.selected_tools = []
        result = validator.validate(plan, registry)
        assert result.is_valid is False
        codes = {e.code for e in result.errors}
        assert "NO_TOOLS_SELECTED" in codes

    def test_missing_tool_in_registry_is_invalid(
        self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry
    ):
        plan = _build_valid_plan(builder)
        plan.selected_tools = ["phantom_tool"]
        result = validator.validate(plan, registry)
        assert result.is_valid is False

    def test_missing_tool_sets_missing_tool_error_code(
        self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry
    ):
        plan = _build_valid_plan(builder)
        plan.selected_tools = ["phantom_tool"]
        result = validator.validate(plan, registry)
        codes = {e.code for e in result.errors}
        assert "MISSING_TOOL" in codes


# ---------------------------------------------------------------------------
# Pattern validation
# ---------------------------------------------------------------------------


class TestValidatorPattern:
    def test_empty_pattern_is_invalid(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        plan = _build_valid_plan(builder)
        plan.selected_pattern = ""
        result = validator.validate(plan, registry)
        assert result.is_valid is False
        codes = {e.code for e in result.errors}
        assert "NO_PATTERN_SELECTED" in codes

    def test_unregistered_pattern_is_invalid(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        plan = _build_valid_plan(builder)
        plan.selected_pattern = "nonexistent_pattern"
        result = validator.validate(plan, registry)
        assert result.is_valid is False
        codes = {e.code for e in result.errors}
        assert "MISSING_PATTERN" in codes


# ---------------------------------------------------------------------------
# Capability coverage
# ---------------------------------------------------------------------------


class TestValidatorCapabilityCoverage:
    def test_unmatched_capability_produces_warning(
        self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry
    ):
        analysis = _analysis(required_capabilities=["fetch_pr_data", "telepathy"])
        plan = builder.build("Review PR #42", analysis)
        result = validator.validate(plan, registry)
        codes = {w.code for w in result.warnings}
        assert "CAPABILITY_UNMATCHED" in codes

    def test_all_standard_capabilities_matched_no_unmatched_warning(
        self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry
    ):
        plan = _build_valid_plan(builder)
        result = validator.validate(plan, registry)
        codes = {w.code for w in result.warnings}
        assert "CAPABILITY_UNMATCHED" not in codes


# ---------------------------------------------------------------------------
# HITL recommendations
# ---------------------------------------------------------------------------


class TestValidatorHITL:
    def test_high_risk_without_hitl_produces_warning(
        self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry
    ):
        plan = builder.build(
            "Review production PR",
            _analysis(risk_level=RiskLevel.HIGH, requires_hitl=False),
        )
        result = validator.validate(plan, registry)
        codes = {w.code for w in result.warnings}
        assert "HITL_RECOMMENDED" in codes

    def test_critical_risk_without_hitl_produces_warning(
        self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry
    ):
        plan = builder.build(
            "Review critical PR",
            _analysis(risk_level=RiskLevel.CRITICAL, requires_hitl=False),
        )
        result = validator.validate(plan, registry)
        codes = {w.code for w in result.warnings}
        assert "HITL_RECOMMENDED" in codes

    def test_high_risk_with_hitl_no_hitl_warning(
        self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry
    ):
        plan = builder.build(
            "Review production PR",
            _analysis(risk_level=RiskLevel.HIGH, requires_hitl=True),
        )
        result = validator.validate(plan, registry)
        codes = {w.code for w in result.warnings}
        assert "HITL_RECOMMENDED" not in codes

    def test_low_risk_without_hitl_no_hitl_warning(
        self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry
    ):
        plan = builder.build("Review PR #42", _analysis(risk_level=RiskLevel.LOW))
        result = validator.validate(plan, registry)
        codes = {w.code for w in result.warnings}
        assert "HITL_RECOMMENDED" not in codes

    def test_very_low_confidence_without_hitl_produces_hitl_warning(
        self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry
    ):
        plan = builder.build("Unclear PR goal", _analysis(confidence=0.4, requires_hitl=False))
        result = validator.validate(plan, registry)
        codes = {w.code for w in result.warnings}
        assert "HITL_RECOMMENDED_LOW_CONFIDENCE" in codes


# ---------------------------------------------------------------------------
# MISSING_CAPABILITIES (Phase D)
# ---------------------------------------------------------------------------


class TestValidatorMissingCapabilities:
    def test_missing_capabilities_is_invalid(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        analysis = _analysis(
            required_capabilities=["fetch_pr_data", "review_code_quality"],
            missing_capabilities=["unknown_cap"],
        )
        plan = builder.build("Review PR", analysis)
        result = validator.validate(plan, registry)
        assert result.is_valid is False

    def test_missing_capabilities_error_code(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        analysis = _analysis(
            required_capabilities=["fetch_pr_data"],
            missing_capabilities=["slack_notify", "send_email"],
        )
        plan = builder.build("Review PR with notifications", analysis)
        result = validator.validate(plan, registry)
        codes = {e.code for e in result.errors}
        assert "MISSING_CAPABILITIES" in codes

    def test_missing_capabilities_message_names_the_caps(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        analysis = _analysis(
            required_capabilities=["fetch_pr_data"],
            missing_capabilities=["slack_notify"],
        )
        plan = builder.build("Review PR", analysis)
        result = validator.validate(plan, registry)
        missing_errors = [e for e in result.errors if e.code == "MISSING_CAPABILITIES"]
        assert len(missing_errors) == 1
        assert "slack_notify" in missing_errors[0].message

    def test_missing_capabilities_does_not_short_circuit(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        """MISSING_CAPABILITIES is blocking but does not skip other checks."""
        analysis = _analysis(
            required_capabilities=["fetch_pr_data"],
            missing_capabilities=["unknown_cap"],
        )
        plan = builder.build("Review PR", analysis)
        # Force an additional agent error to confirm validation continued
        plan.selected_agents = ["nonexistent_agent"]
        result = validator.validate(plan, registry)
        codes = {e.code for e in result.errors}
        assert "MISSING_CAPABILITIES" in codes
        assert "MISSING_AGENT" in codes

    def test_empty_missing_capabilities_no_error(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        analysis = _analysis(missing_capabilities=[])
        plan = builder.build("Review PR", analysis)
        result = validator.validate(plan, registry)
        codes = {e.code for e in result.errors}
        assert "MISSING_CAPABILITIES" not in codes


# ---------------------------------------------------------------------------
# DATAFLOW_UNSATISFIED (Phase D)
# ---------------------------------------------------------------------------


class TestValidatorDataflow:
    def test_full_pr_review_dataflow_passes(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        """All four PR review agents selected → full dataflow satisfied."""
        plan = _build_valid_plan(builder)
        # Confirm all four agents are present
        assert set(plan.selected_agents) == {
            "pr_data_agent", "review_specialist", "risk_specialist", "synthesis_agent"
        }
        result = validator.validate(plan, registry)
        error_codes = {e.code for e in result.errors}
        assert "DATAFLOW_UNSATISFIED" not in error_codes

    def test_missing_producer_triggers_dataflow_error(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        """synthesis_agent without review_specialist → code_quality_report unsatisfied."""
        plan = _build_valid_plan(builder)
        plan.selected_agents = ["pr_data_agent", "risk_specialist", "synthesis_agent"]
        result = validator.validate(plan, registry)
        error_codes = {e.code for e in result.errors}
        assert "DATAFLOW_UNSATISFIED" in error_codes

    def test_dataflow_error_names_missing_token(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        plan = _build_valid_plan(builder)
        plan.selected_agents = ["pr_data_agent", "risk_specialist", "synthesis_agent"]
        result = validator.validate(plan, registry)
        df_errors = [e for e in result.errors if e.code == "DATAFLOW_UNSATISFIED"]
        assert any("code_quality_report" in e.message for e in df_errors)

    def test_agents_without_consumes_do_not_fail_dataflow(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        """pr_data_agent has consumes=[] — nothing to check, no DATAFLOW error."""
        analysis = _analysis(required_capabilities=["fetch_pr_data", "fetch_github_diff"])
        plan = builder.build("Fetch PR data only", analysis)
        assert plan.selected_agents == ["pr_data_agent"]
        result = validator.validate(plan, registry)
        error_codes = {e.code for e in result.errors}
        assert "DATAFLOW_UNSATISFIED" not in error_codes

    def test_user_provided_tokens_not_validated(self, validator: PlanValidator):
        """Tokens not produced by any registered agent are user input — no DATAFLOW error."""
        reg = CapabilityRegistry()
        reg.register_agent(AgentCapabilityDescriptor(
            agent_id="worker",
            name="Worker",
            description="",
            capabilities=["do_work"],
            consumes=["raw_user_data"],  # no registered agent produces this
            produces=["processed_result"],
        ))
        reg.register_pattern(PatternCapabilityDescriptor(
            pattern="parallel_specialist",
            name="Parallel Specialist",
            description="",
            best_for=[],
            supported_task_types=["code_review"],
            trigger_capabilities=[],
        ))
        analysis = _analysis(required_capabilities=["do_work"])
        plan = _make_plan(analysis, agents=["worker"], pattern="parallel_specialist")
        result = validator.validate(plan, reg)
        error_codes = {e.code for e in result.errors}
        assert "DATAFLOW_UNSATISFIED" not in error_codes

    def test_dataflow_satisfied_when_pr_diff_producer_selected(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        """review_specialist consumes pr_diff; pr_data_agent produces it — satisfied."""
        plan = _build_valid_plan(builder)
        plan.selected_agents = ["pr_data_agent", "review_specialist"]
        result = validator.validate(plan, registry)
        error_codes = {e.code for e in result.errors}
        assert "DATAFLOW_UNSATISFIED" not in error_codes

    def test_all_consumers_missing_producers_reported(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        """synthesis_agent missing both specialist producers — two unsatisfied tokens."""
        plan = _build_valid_plan(builder)
        # Only pr_data_agent + synthesis_agent: neither specialist is present
        plan.selected_agents = ["pr_data_agent", "synthesis_agent"]
        result = validator.validate(plan, registry)
        df_errors = [e for e in result.errors if e.code == "DATAFLOW_UNSATISFIED"]
        # Both code_quality_report and risk_assessment_report are unsatisfied
        assert len(df_errors) == 2
        missing_tokens = {e.message.split("'")[1] for e in df_errors}
        assert "code_quality_report" in missing_tokens
        assert "risk_assessment_report" in missing_tokens

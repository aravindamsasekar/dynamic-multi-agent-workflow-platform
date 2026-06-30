"""Unit tests for PlanValidator — deterministic, no LLM."""

from __future__ import annotations

import pytest

from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.models import (
    GoalAnalysis,
    GeneratedWorkflowPlan,
    GuardrailConfig,
    RiskLevel,
    TaskType,
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
    task_type: TaskType = TaskType.CODE_REVIEW,
    confidence: float = 0.92,
    risk_level: RiskLevel = RiskLevel.LOW,
    requires_hitl: bool = False,
    required_capabilities: list[str] | None = None,
    constraints: list[str] | None = None,
) -> GoalAnalysis:
    return GoalAnalysis(
        task_type=task_type,
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
        task_type=TaskType.UNSUPPORTED,
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
# Task type validation
# ---------------------------------------------------------------------------


class TestValidatorTaskType:
    def test_unsupported_task_type_is_invalid(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        plan = builder.build("Random goal", _unsupported_analysis())
        result = validator.validate(plan, registry)
        assert result.is_valid is False

    def test_unsupported_sets_error_code(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        plan = builder.build("Random goal", _unsupported_analysis())
        result = validator.validate(plan, registry)
        codes = {e.code for e in result.errors}
        assert "UNSUPPORTED_TASK_TYPE" in codes

    def test_unsupported_short_circuits_other_checks(self, builder: PlanBuilder, validator: PlanValidator, registry: CapabilityRegistry):
        plan = builder.build("Random goal", _unsupported_analysis())
        result = validator.validate(plan, registry)
        # Only one error — UNSUPPORTED_TASK_TYPE — no additional agent/tool errors
        assert len(result.errors) == 1


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

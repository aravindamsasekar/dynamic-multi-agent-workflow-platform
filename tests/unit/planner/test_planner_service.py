"""Unit tests for PlannerService — mocks GoalAnalyzer's LLM call."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.models import (
    GoalAnalysis,
    RiskLevel,
    TaskType,
    ValidationResult,
)
from platform.planner.planner_service import PlannerService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> CapabilityRegistry:
    return CapabilityRegistry.build_pr_review_registry()


@pytest.fixture
def mock_llm():
    return MagicMock()


def _mock_analysis(
    task_type: TaskType = TaskType.CODE_REVIEW,
    confidence: float = 0.9,
    risk_level: RiskLevel = RiskLevel.LOW,
    requires_hitl: bool = False,
) -> GoalAnalysis:
    return GoalAnalysis(
        task_type=task_type,
        required_capabilities=["fetch_pr_data", "review_code_quality", "synthesize_findings"],
        risk_level=risk_level,
        confidence=confidence,
        reasoning="Classified as code review.",
        constraints=["read_only"],
        requires_hitl=requires_hitl,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_generate_returns_plan_and_validation(registry, mock_llm):
    service = PlannerService(llm=mock_llm, registry=registry)
    analysis = _mock_analysis()

    with patch.object(service._analyzer, "analyze", new=AsyncMock(return_value=analysis)):
        plan, validation = await service.generate("Review my PR at github.com/org/repo/pull/42")

    assert plan.user_goal == "Review my PR at github.com/org/repo/pull/42"
    assert plan.plan_id  # non-empty UUID
    assert plan.selected_pattern == "parallel_specialist"
    assert len(plan.selected_agents) > 0
    assert len(plan.selected_tools) > 0
    assert isinstance(validation, ValidationResult)


async def test_generate_plan_id_is_unique(registry, mock_llm):
    service = PlannerService(llm=mock_llm, registry=registry)
    analysis = _mock_analysis()

    with patch.object(service._analyzer, "analyze", new=AsyncMock(return_value=analysis)):
        plan1, _ = await service.generate("Goal A")
        plan2, _ = await service.generate("Goal B")

    assert plan1.plan_id != plan2.plan_id


async def test_generate_valid_code_review_plan(registry, mock_llm):
    service = PlannerService(llm=mock_llm, registry=registry)
    analysis = _mock_analysis()

    with patch.object(service._analyzer, "analyze", new=AsyncMock(return_value=analysis)):
        plan, validation = await service.generate("Review PR #42")

    assert validation.is_valid
    assert len(validation.errors) == 0


async def test_generate_includes_estimated_fields(registry, mock_llm):
    service = PlannerService(llm=mock_llm, registry=registry)
    analysis = _mock_analysis()

    with patch.object(service._analyzer, "analyze", new=AsyncMock(return_value=analysis)):
        plan, _ = await service.generate("Review PR #42")

    assert plan.estimated_complexity in ("low", "medium", "high")
    assert plan.estimated_duration_seconds > 0


async def test_generate_high_risk_requires_hitl(registry, mock_llm):
    service = PlannerService(llm=mock_llm, registry=registry)
    analysis = _mock_analysis(risk_level=RiskLevel.HIGH, requires_hitl=True)

    with patch.object(service._analyzer, "analyze", new=AsyncMock(return_value=analysis)):
        plan, _ = await service.generate("Review high-risk PR")

    assert plan.hitl_required is True


async def test_generate_passes_goal_to_analyzer(registry, mock_llm):
    service = PlannerService(llm=mock_llm, registry=registry)
    analysis = _mock_analysis()
    goal = "Please review pull request #99 in my repo"

    mock_analyze = AsyncMock(return_value=analysis)
    with patch.object(service._analyzer, "analyze", new=mock_analyze):
        await service.generate(goal)

    mock_analyze.assert_called_once_with(goal)


async def test_generate_unsupported_goal_is_invalid(registry, mock_llm):
    service = PlannerService(llm=mock_llm, registry=registry)
    analysis = GoalAnalysis(
        task_type=TaskType.UNSUPPORTED,
        required_capabilities=[],
        risk_level=RiskLevel.LOW,
        confidence=0.3,
        reasoning="Unsupported task.",
        constraints=[],
        requires_hitl=False,
    )

    with patch.object(service._analyzer, "analyze", new=AsyncMock(return_value=analysis)):
        plan, validation = await service.generate("Deploy my app to production")

    assert not validation.is_valid
    assert any(e.code == "UNSUPPORTED_TASK_TYPE" for e in validation.errors)


# ---------------------------------------------------------------------------
# Planner error propagation
# ---------------------------------------------------------------------------


async def test_generate_propagates_planner_error(registry, mock_llm):
    from platform.planner.models import PlannerError

    service = PlannerService(llm=mock_llm, registry=registry)

    with patch.object(
        service._analyzer,
        "analyze",
        new=AsyncMock(side_effect=PlannerError("LLM returned garbage")),
    ):
        with pytest.raises(PlannerError, match="LLM returned garbage"):
            await service.generate("Anything")


# ---------------------------------------------------------------------------
# Plan content checks
# ---------------------------------------------------------------------------


async def test_generate_explanation_is_non_empty(registry, mock_llm):
    service = PlannerService(llm=mock_llm, registry=registry)
    analysis = _mock_analysis()

    with patch.object(service._analyzer, "analyze", new=AsyncMock(return_value=analysis)):
        plan, _ = await service.generate("Review PR")

    assert len(plan.explanation) > 0


async def test_generate_goal_analysis_stored_on_plan(registry, mock_llm):
    service = PlannerService(llm=mock_llm, registry=registry)
    analysis = _mock_analysis(confidence=0.77)

    with patch.object(service._analyzer, "analyze", new=AsyncMock(return_value=analysis)):
        plan, _ = await service.generate("Review PR")

    assert plan.goal_analysis.confidence == pytest.approx(0.77)
    assert plan.goal_analysis.task_type == TaskType.CODE_REVIEW


async def test_generate_low_confidence_produces_validation_warning(registry, mock_llm):
    service = PlannerService(llm=mock_llm, registry=registry)
    analysis = _mock_analysis(confidence=0.65)

    with patch.object(service._analyzer, "analyze", new=AsyncMock(return_value=analysis)):
        plan, validation = await service.generate("Review PR")

    warning_codes = {w.code for w in validation.warnings}
    assert "LOW_CONFIDENCE" in warning_codes or "HITL_RECOMMENDED_LOW_CONFIDENCE" in warning_codes


async def test_generate_medium_complexity(registry, mock_llm):
    """3-4 agents → medium complexity."""
    service = PlannerService(llm=mock_llm, registry=registry)
    analysis = _mock_analysis()

    with patch.object(service._analyzer, "analyze", new=AsyncMock(return_value=analysis)):
        plan, _ = await service.generate("Review PR")

    # PR review registry has 4 agents → medium
    assert plan.estimated_complexity == "medium"

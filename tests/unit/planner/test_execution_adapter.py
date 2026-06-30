"""Unit tests for ExecutionAdapter — mocks Orchestrator and WorkflowRegistry."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from platform.core.models.workflow import PatternType, RunStatus, WorkflowDefinition, WorkflowResult
from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.execution_adapter import ExecutionAdapter
from platform.planner.models import (
    GeneratedWorkflowPlan,
    GoalAnalysis,
    GuardrailConfig,
    RiskLevel,
    TaskType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> CapabilityRegistry:
    return CapabilityRegistry.build_pr_review_registry()


@pytest.fixture
def mock_orchestrator():
    orch = MagicMock()
    orch.run = AsyncMock(
        return_value=WorkflowResult(
            run_id="run-abc-123",
            workflow_id="plan-xyz",
            output="PR looks good.",
            status=RunStatus.COMPLETED,
        )
    )
    return orch


@pytest.fixture
def mock_workflow_registry():
    reg = MagicMock()
    reg.register = MagicMock()
    return reg


@pytest.fixture
def adapter(mock_orchestrator, mock_workflow_registry, registry) -> ExecutionAdapter:
    return ExecutionAdapter(
        orchestrator=mock_orchestrator,
        workflow_registry=mock_workflow_registry,
        capability_registry=registry,
    )


def _make_plan(
    plan_id: str = "plan-xyz",
    pattern: str = "parallel_specialist",
    agents: list[str] | None = None,
    hitl_required: bool = False,
) -> GeneratedWorkflowPlan:
    return GeneratedWorkflowPlan(
        plan_id=plan_id,
        user_goal="Review PR #42",
        goal_analysis=GoalAnalysis(
            task_type=TaskType.CODE_REVIEW,
            required_capabilities=["fetch_pr_data", "review_code_quality", "synthesize_findings"],
            risk_level=RiskLevel.LOW,
            confidence=0.9,
            reasoning="Code review task.",
            constraints=["read_only"],
            requires_hitl=hitl_required,
        ),
        selected_pattern=pattern,
        selected_agents=agents if agents is not None else ["pr_data_agent", "review_specialist", "risk_specialist", "synthesis_agent"],
        selected_tools=["github_get_pr", "github_get_diff", "knowledge_search"],
        guardrails=[GuardrailConfig(rule_type="content_filter", config={}, reason="safety")],
        hitl_required=hitl_required,
        warnings=[],
        explanation="Parallel PR review workflow.",
        estimated_complexity="medium",
        estimated_duration_seconds=75,
    )


# ---------------------------------------------------------------------------
# to_workflow_definition
# ---------------------------------------------------------------------------


def test_to_workflow_definition_pattern_type(adapter):
    plan = _make_plan()
    wf = adapter.to_workflow_definition(plan)
    assert wf.pattern == PatternType.PARALLEL_SPECIALIST


def test_to_workflow_definition_workflow_id_matches_plan_id(adapter):
    plan = _make_plan(plan_id="plan-unique-99")
    wf = adapter.to_workflow_definition(plan)
    assert wf.workflow_id == "plan-unique-99"


def test_to_workflow_definition_agent_ids_preserved(adapter):
    plan = _make_plan()
    wf = adapter.to_workflow_definition(plan)
    assert wf.agent_ids == plan.selected_agents


def test_to_workflow_definition_parallel_has_reviewer(adapter):
    plan = _make_plan()
    wf = adapter.to_workflow_definition(plan)
    assert "reviewer_agent_id" in wf.pattern_config
    assert wf.pattern_config["reviewer_agent_id"] == "synthesis_agent"


def test_to_workflow_definition_parallel_has_strategy(adapter):
    plan = _make_plan()
    wf = adapter.to_workflow_definition(plan)
    assert wf.pattern_config.get("strategy") == "concatenate"


def test_to_workflow_definition_hitl_enabled_when_required(adapter):
    plan = _make_plan(hitl_required=True)
    wf = adapter.to_workflow_definition(plan)
    assert wf.hitl_enabled is True


def test_to_workflow_definition_hitl_disabled_when_not_required(adapter):
    plan = _make_plan(hitl_required=False)
    wf = adapter.to_workflow_definition(plan)
    assert wf.hitl_enabled is False


def test_to_workflow_definition_name_contains_goal(adapter):
    plan = _make_plan()
    wf = adapter.to_workflow_definition(plan)
    assert "Review PR #42" in wf.name


def test_to_workflow_definition_description_is_explanation(adapter):
    plan = _make_plan()
    wf = adapter.to_workflow_definition(plan)
    assert wf.description == plan.explanation


def test_to_workflow_definition_no_reviewer_fallback(adapter):
    """When no agent has 'synthesize_findings', falls back to last agent."""
    plan = _make_plan(agents=["pr_data_agent", "review_specialist"])
    wf = adapter.to_workflow_definition(plan)
    assert wf.pattern_config.get("reviewer_agent_id") == "review_specialist"


def test_to_workflow_definition_returns_workflow_definition(adapter):
    plan = _make_plan()
    wf = adapter.to_workflow_definition(plan)
    assert isinstance(wf, WorkflowDefinition)


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


async def test_execute_registers_workflow(adapter, mock_workflow_registry):
    plan = _make_plan()
    await adapter.execute(plan, "PR input")
    mock_workflow_registry.register.assert_called_once()
    registered_wf = mock_workflow_registry.register.call_args[0][0]
    assert registered_wf.workflow_id == plan.plan_id


async def test_execute_calls_orchestrator_with_plan_id(adapter, mock_orchestrator):
    plan = _make_plan(plan_id="plan-execute-me")
    await adapter.execute(plan, "some input")
    mock_orchestrator.run.assert_called_once_with("plan-execute-me", "some input")


async def test_execute_returns_workflow_result(adapter):
    plan = _make_plan()
    result = await adapter.execute(plan, "some input")
    assert isinstance(result, WorkflowResult)
    assert result.run_id == "run-abc-123"
    assert result.output == "PR looks good."


async def test_execute_passes_dict_input(adapter, mock_orchestrator):
    plan = _make_plan()
    input_data = {"owner": "org", "repo": "myrepo", "pr_number": 42}
    await adapter.execute(plan, input_data)
    mock_orchestrator.run.assert_called_once_with(plan.plan_id, input_data)


async def test_execute_registers_before_calling_orchestrator(adapter, mock_workflow_registry, mock_orchestrator):
    """Ensure register() is called before orchestrator.run()."""
    call_order = []
    mock_workflow_registry.register.side_effect = lambda *a, **kw: call_order.append("register")
    mock_orchestrator.run.side_effect = AsyncMock(
        side_effect=lambda *a, **kw: call_order.append("run") or WorkflowResult(
            run_id="r", workflow_id="w", output="ok", status=RunStatus.COMPLETED
        )
    )

    plan = _make_plan()
    await adapter.execute(plan, "input")

    assert call_order == ["register", "run"]


# ---------------------------------------------------------------------------
# Reviewer detection
# ---------------------------------------------------------------------------


def test_find_reviewer_id_returns_synthesis_agent(adapter, registry):
    plan = _make_plan()
    reviewer = adapter._find_reviewer_id(plan)
    assert reviewer == "synthesis_agent"


def test_find_reviewer_id_fallback_to_last_agent(adapter, registry):
    plan = _make_plan(agents=["pr_data_agent", "review_specialist"])
    reviewer = adapter._find_reviewer_id(plan)
    assert reviewer == "review_specialist"


def test_find_reviewer_id_none_when_no_agents(adapter, registry):
    plan = _make_plan(agents=[])
    reviewer = adapter._find_reviewer_id(plan)
    assert reviewer is None

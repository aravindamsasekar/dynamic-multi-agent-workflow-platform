"""Unit tests for ExecutionAdapter — mocks Orchestrator and WorkflowRegistry."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from platform.core.models.agent import AgentDefinition, LLMConfig
from platform.core.models.workflow import PatternType, RunStatus, WorkflowDefinition, WorkflowResult
from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.execution_adapter import ExecutionAdapter
from platform.planner.models import (
    GeneratedWorkflowPlan,
    GoalAnalysis,
    GuardrailConfig,
    RiskLevel,
    RuntimeAgentDefinition,
)
from platform.registries.agent_registry import AgentRegistry


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
def mock_agent_registry():
    reg = MagicMock()
    reg.exists = MagicMock(return_value=False)
    reg.register = MagicMock()
    reg.unregister = MagicMock()
    return reg


@pytest.fixture
def adapter(mock_orchestrator, mock_workflow_registry, mock_agent_registry, registry) -> ExecutionAdapter:
    return ExecutionAdapter(
        orchestrator=mock_orchestrator,
        workflow_registry=mock_workflow_registry,
        agent_registry=mock_agent_registry,
        capability_registry=registry,
    )


def _make_plan(
    plan_id: str = "plan-xyz",
    pattern: str = "parallel_specialist",
    agents: list[str] | None = None,
    hitl_required: bool = False,
    runtime_agents: list[RuntimeAgentDefinition] | None = None,
) -> GeneratedWorkflowPlan:
    selected = agents if agents is not None else ["pr_data_agent", "review_specialist", "risk_specialist", "synthesis_agent"]
    return GeneratedWorkflowPlan(
        plan_id=plan_id,
        user_goal="Review PR #42",
        goal_analysis=GoalAnalysis(
            required_capabilities=["fetch_pr_data", "review_code_quality", "synthesize_findings"],
            risk_level=RiskLevel.LOW,
            confidence=0.9,
            reasoning="Code review task.",
            constraints=["read_only"],
            requires_hitl=hitl_required,
        ),
        selected_pattern=pattern,
        selected_agents=selected,
        runtime_agents=runtime_agents if runtime_agents is not None else [],
        selected_tools=["github_get_pr", "github_get_diff", "knowledge_search"],
        guardrails=[GuardrailConfig(rule_type="content_filter", config={}, reason="safety")],
        hitl_required=hitl_required,
        warnings=[],
        explanation="Parallel PR review workflow.",
        estimated_complexity="medium",
        estimated_duration_seconds=75,
    )


def _make_runtime_agent(
    agent_id: str,
    capabilities: list[str],
    generated: bool = False,
    tool_names: list[str] | None = None,
    system_prompt: str = "",
) -> RuntimeAgentDefinition:
    return RuntimeAgentDefinition(
        id=agent_id,
        name=agent_id.replace("_", " ").title(),
        description=f"{agent_id} description.",
        capabilities=capabilities,
        tool_names=tool_names or [],
        system_prompt=system_prompt if generated else "",
        generated=generated,
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
    # runtime_agents=[] (old plan) → fallback to selected_agents
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


def test_to_workflow_definition_no_reviewer_when_no_synthesizer(adapter):
    """When no agent has 'synthesize_findings', no reviewer is added."""
    plan = _make_plan(agents=["pr_data_agent", "review_specialist"])
    wf = adapter.to_workflow_definition(plan)
    assert "reviewer_agent_id" not in wf.pattern_config


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
# Goal fallback — empty input_data uses plan.user_goal
# ---------------------------------------------------------------------------


def _make_fs_plan(plan_id: str = "fs-plan-001") -> GeneratedWorkflowPlan:
    """Plan with a single generated filesystem_read agent, goal = 'Read README.md'."""
    gen = _make_runtime_agent(
        f"gen_{plan_id}_filesystem_read",
        ["filesystem_read"],
        generated=True,
        tool_names=["filesystem_read_file"],
        system_prompt="You execute filesystem_read capability.",
    )
    return GeneratedWorkflowPlan(
        plan_id=plan_id,
        user_goal="Read README.md",
        goal_analysis=GoalAnalysis(
            required_capabilities=["filesystem_read"],
            risk_level=RiskLevel.LOW,
            confidence=0.9,
            reasoning="File read task.",
            constraints=["read_only"],
            requires_hitl=False,
        ),
        selected_pattern="parallel_specialist",
        selected_agents=[],
        runtime_agents=[gen],
        selected_tools=["filesystem_read_file"],
        guardrails=[],
        hitl_required=False,
        warnings=[],
        explanation="Generated filesystem read workflow.",
        estimated_complexity="low",
        estimated_duration_seconds=25,
    )


async def test_execute_uses_plan_goal_when_input_is_empty_dict(adapter, mock_orchestrator):
    """Regression: empty approve body must not produce an empty agent task.

    When input_data={} the orchestrator must receive plan.user_goal, not '{}',
    so generated agents have a meaningful user message.
    """
    plan = _make_fs_plan()
    await adapter.execute(plan, {})
    mock_orchestrator.run.assert_called_once_with(plan.plan_id, "Read README.md")


async def test_execute_uses_plan_goal_when_input_is_empty_string(adapter, mock_orchestrator):
    plan = _make_fs_plan()
    await adapter.execute(plan, "")
    mock_orchestrator.run.assert_called_once_with(plan.plan_id, "Read README.md")


async def test_execute_preserves_nonempty_dict_input(adapter, mock_orchestrator):
    """Static PR review: structured payload must NOT be replaced by plan.user_goal."""
    pr_payload = {"owner": "acme", "repo": "app", "pr_number": 42}
    plan = _make_plan()
    await adapter.execute(plan, pr_payload)
    mock_orchestrator.run.assert_called_once_with(plan.plan_id, pr_payload)


async def test_execute_preserves_nonempty_string_input(adapter, mock_orchestrator):
    plan = _make_plan()
    await adapter.execute(plan, "explicit task text")
    mock_orchestrator.run.assert_called_once_with(plan.plan_id, "explicit task text")


async def test_execute_goal_fallback_does_not_affect_static_pr_review(adapter, mock_orchestrator):
    """Static plan with non-empty input is passed through unchanged."""
    plan = _make_plan()
    input_data = {"pr_number": 99}
    await adapter.execute(plan, input_data)
    _, called_input = mock_orchestrator.run.call_args[0]
    assert called_input == input_data
    assert called_input != plan.user_goal


# ---------------------------------------------------------------------------
# Reviewer detection
# ---------------------------------------------------------------------------


def test_find_reviewer_id_returns_synthesis_agent(adapter, registry):
    plan = _make_plan()
    reviewer = adapter._find_reviewer_id(plan)
    assert reviewer == "synthesis_agent"


def test_find_reviewer_id_none_when_no_synthesizer_agent(adapter, registry):
    """No 'synthesize_findings' capability in any selected agent → no reviewer."""
    plan = _make_plan(agents=["pr_data_agent", "review_specialist"])
    reviewer = adapter._find_reviewer_id(plan)
    assert reviewer is None


def test_find_reviewer_id_none_when_no_agents(adapter, registry):
    plan = _make_plan(agents=[])
    reviewer = adapter._find_reviewer_id(plan)
    assert reviewer is None


# ---------------------------------------------------------------------------
# Phase C — generated agent support
# ---------------------------------------------------------------------------


_GEN_ID = "gen-plan-001_filesystem_read"
_GEN_SYSTEM_PROMPT = "You handle filesystem_read."


def _make_mixed_plan(plan_id: str = "gen-plan-001") -> GeneratedWorkflowPlan:
    """Plan with one static + one generated agent in runtime_agents."""
    static = _make_runtime_agent("pr_data_agent", ["fetch_pr_data"], generated=False)
    generated = _make_runtime_agent(
        _GEN_ID, ["filesystem_read"],
        generated=True, tool_names=["fs_tool"], system_prompt=_GEN_SYSTEM_PROMPT,
    )
    return _make_plan(
        plan_id=plan_id,
        agents=["pr_data_agent"],
        runtime_agents=[static, generated],
    )


def _make_all_static_with_runtime(plan_id: str = "static-plan-001") -> GeneratedWorkflowPlan:
    """Plan where all runtime_agents are static (generated=False)."""
    agents = [
        _make_runtime_agent("pr_data_agent", ["fetch_pr_data", "synthesize_findings"]),
        _make_runtime_agent("review_specialist", ["review_code_quality"]),
    ]
    return _make_plan(
        plan_id=plan_id,
        agents=["pr_data_agent", "review_specialist"],
        runtime_agents=agents,
    )


class TestExecutionAdapterPhaseC:
    # --- WorkflowDefinition conversion ---

    def test_to_workflow_definition_uses_runtime_agents_when_populated(self, adapter):
        plan = _make_mixed_plan()
        wf = adapter.to_workflow_definition(plan)
        assert wf.agent_ids == ["pr_data_agent", _GEN_ID]

    def test_to_workflow_definition_includes_generated_agent_id(self, adapter):
        plan = _make_mixed_plan()
        wf = adapter.to_workflow_definition(plan)
        assert _GEN_ID in wf.agent_ids

    def test_to_workflow_definition_all_static_with_runtime_agents(self, adapter):
        plan = _make_all_static_with_runtime()
        wf = adapter.to_workflow_definition(plan)
        assert set(wf.agent_ids) == {"pr_data_agent", "review_specialist"}

    def test_to_workflow_definition_fallback_when_runtime_agents_empty(self, adapter):
        plan = _make_plan(agents=["pr_data_agent", "review_specialist"])
        # runtime_agents=[] → fallback to selected_agents
        wf = adapter.to_workflow_definition(plan)
        assert wf.agent_ids == ["pr_data_agent", "review_specialist"]

    # --- _find_reviewer_id with runtime_agents ---

    def test_find_reviewer_uses_runtime_agent_capabilities(self, adapter):
        synth = _make_runtime_agent("synth_agent", ["synthesize_findings"])
        other = _make_runtime_agent("other_agent", ["review_code_quality"])
        plan = _make_plan(runtime_agents=[other, synth])
        assert adapter._find_reviewer_id(plan) == "synth_agent"

    def test_find_reviewer_none_when_no_synthesize_cap_in_runtime_agents(self, adapter):
        """No agent in runtime_agents has 'synthesize_findings' → None."""
        agents = [
            _make_runtime_agent("pr_data_agent", ["fetch_pr_data"]),
            _make_runtime_agent("review_specialist", ["review_code_quality"]),
        ]
        plan = _make_plan(runtime_agents=agents)
        assert adapter._find_reviewer_id(plan) is None

    def test_find_reviewer_generated_agent_with_synthesize_cap(self, adapter):
        gen_synth = _make_runtime_agent(
            _GEN_ID, ["synthesize_findings"], generated=True
        )
        plan = _make_plan(runtime_agents=[gen_synth])
        assert adapter._find_reviewer_id(plan) == _GEN_ID

    # --- _to_agent_definition ---

    def test_to_agent_definition_maps_all_fields(self):
        agent = _make_runtime_agent(
            _GEN_ID, ["filesystem_read"],
            generated=True, tool_names=["fs_tool"], system_prompt=_GEN_SYSTEM_PROMPT,
        )
        defn = ExecutionAdapter._to_agent_definition(agent)
        assert isinstance(defn, AgentDefinition)
        assert defn.agent_id == _GEN_ID
        assert defn.name == agent.name
        assert defn.description == agent.description
        assert defn.system_prompt == _GEN_SYSTEM_PROMPT
        assert defn.tool_names == ["fs_tool"]
        assert isinstance(defn.llm_config, LLMConfig)

    def test_to_agent_definition_uses_default_llm_config(self):
        agent = _make_runtime_agent(_GEN_ID, [], generated=True, system_prompt="x")
        defn = ExecutionAdapter._to_agent_definition(agent)
        assert defn.llm_config == LLMConfig()

    # --- execute() registration / cleanup ---

    async def test_execute_registers_generated_agents_before_run(
        self, mock_orchestrator, mock_workflow_registry, registry
    ):
        """Generated agents must be in AgentRegistry when orchestrator.run() is called."""
        real_registry = AgentRegistry()
        adapter = ExecutionAdapter(
            orchestrator=mock_orchestrator,
            workflow_registry=mock_workflow_registry,
            agent_registry=real_registry,
            capability_registry=registry,
        )
        ids_during_run: list[str] = []
        async def _capture_run(workflow_id, input_data):
            ids_during_run.append(_GEN_ID)
            assert real_registry.exists(_GEN_ID), "generated agent must be registered during run"
            return WorkflowResult(
                run_id="r", workflow_id="w", output="ok", status=RunStatus.COMPLETED
            )
        mock_orchestrator.run.side_effect = _capture_run

        await adapter.execute(_make_mixed_plan(), "input")
        assert _GEN_ID in ids_during_run

    async def test_execute_cleans_up_generated_agents_on_success(
        self, mock_orchestrator, mock_workflow_registry, registry
    ):
        real_registry = AgentRegistry()
        adapter = ExecutionAdapter(
            orchestrator=mock_orchestrator,
            workflow_registry=mock_workflow_registry,
            agent_registry=real_registry,
            capability_registry=registry,
        )
        await adapter.execute(_make_mixed_plan(), "input")
        assert not real_registry.exists(_GEN_ID)

    async def test_execute_cleans_up_generated_agents_on_failure(
        self, mock_orchestrator, mock_workflow_registry, registry
    ):
        real_registry = AgentRegistry()
        mock_orchestrator.run.side_effect = RuntimeError("orchestrator failed")
        adapter = ExecutionAdapter(
            orchestrator=mock_orchestrator,
            workflow_registry=mock_workflow_registry,
            agent_registry=real_registry,
            capability_registry=registry,
        )
        with pytest.raises(RuntimeError, match="orchestrator failed"):
            await adapter.execute(_make_mixed_plan(), "input")
        assert not real_registry.exists(_GEN_ID)

    async def test_execute_does_not_register_static_agents(self, adapter, mock_agent_registry):
        plan = _make_all_static_with_runtime()
        await adapter.execute(plan, "input")
        mock_agent_registry.register.assert_not_called()

    async def test_execute_no_runtime_agents_no_registration(self, adapter, mock_agent_registry):
        # Old plan with runtime_agents=[] — no registration
        plan = _make_plan()
        await adapter.execute(plan, "input")
        mock_agent_registry.register.assert_not_called()

    async def test_execute_skips_registration_if_id_already_exists(
        self, mock_orchestrator, mock_workflow_registry, registry
    ):
        """ID collision: existing agent not overwritten, not in cleanup list."""
        real_registry = AgentRegistry()
        # Pre-register with a different definition
        real_registry.register(AgentDefinition(
            agent_id=_GEN_ID, name="Pre-existing", description="", system_prompt="pre"
        ))
        adapter = ExecutionAdapter(
            orchestrator=mock_orchestrator,
            workflow_registry=mock_workflow_registry,
            agent_registry=real_registry,
            capability_registry=registry,
        )
        await adapter.execute(_make_mixed_plan(), "input")
        # Pre-existing agent must not have been overwritten
        assert real_registry.get(_GEN_ID).name == "Pre-existing"
        # And must still exist after cleanup (not unregistered since we didn't register it)
        assert real_registry.exists(_GEN_ID)

    async def test_execute_unregister_called_once_per_registered_agent(
        self, adapter, mock_agent_registry
    ):
        plan = _make_mixed_plan()
        await adapter.execute(plan, "input")
        mock_agent_registry.unregister.assert_called_once_with(_GEN_ID)

    async def test_execute_does_not_unregister_static_agents(
        self, adapter, mock_agent_registry
    ):
        plan = _make_all_static_with_runtime()
        await adapter.execute(plan, "input")
        mock_agent_registry.unregister.assert_not_called()

    # --- generated agent without synthesize_findings must not become reviewer ---

    def test_single_generated_agent_no_synthesize_cap_has_no_reviewer(self, adapter):
        """filesystem_read agent (no synthesize_findings) must NOT be its own reviewer."""
        gen_agent = _make_runtime_agent(
            _GEN_ID, ["filesystem_read"], generated=True,
            tool_names=["filesystem_read_file"], system_prompt=_GEN_SYSTEM_PROMPT,
        )
        plan = _make_plan(
            plan_id="gen-plan-001",
            pattern="parallel_specialist",
            agents=[],
            runtime_agents=[gen_agent],
        )
        wf = adapter.to_workflow_definition(plan)
        assert "reviewer_agent_id" not in wf.pattern_config

    def test_find_reviewer_none_for_single_generated_agent_without_synthesis(self, adapter):
        gen_agent = _make_runtime_agent(
            _GEN_ID, ["filesystem_read"], generated=True,
        )
        plan = _make_plan(runtime_agents=[gen_agent])
        assert adapter._find_reviewer_id(plan) is None

    def test_workflow_def_for_generated_only_plan_has_no_reviewer_agent_id(self, adapter):
        """Regression: generated filesystem_read agent must not appear as reviewer_agent_id."""
        gen_agent = _make_runtime_agent(
            _GEN_ID, ["filesystem_read"], generated=True,
            tool_names=["filesystem_read_file"],
        )
        plan = _make_plan(
            plan_id="gen-plan-001",
            agents=[],
            runtime_agents=[gen_agent],
        )
        wf = adapter.to_workflow_definition(plan)
        # Ensure the generated agent ID is NOT the reviewer
        assert wf.pattern_config.get("reviewer_agent_id") != _GEN_ID

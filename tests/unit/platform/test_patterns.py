"""Unit tests for pattern executors."""

from __future__ import annotations

import pytest

from platform.core.exceptions import AgentNotFound, PatternExecutionError
from platform.core.interfaces.observer import IObserver
from platform.core.models.agent import AgentDefinition
from platform.core.models.context import ExecutionContext
from platform.core.models.events import WorkflowEvent
from platform.core.models.message import LLMResponse, Role, StopReason, TextContent
from platform.core.models.workflow import PatternType, WorkflowDefinition, WorkflowResult
from platform.llm.mock_provider import MockLLMProvider
from platform.memory.in_memory_store import InMemoryStore
from platform.patterns.parallel_specialist import ParallelSpecialistExecutor
from platform.patterns.router import RouterExecutor
from platform.policy.engine import PolicyEngine
from platform.registries.agent_registry import AgentRegistry
from platform.registries.tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_RUN_ID = "test-run-001"


class _CapturingObserver(IObserver):
    def __init__(self) -> None:
        self.events: list[WorkflowEvent] = []

    def on_event(self, event: WorkflowEvent) -> None:
        self.events.append(event)


def _response(text: str) -> LLMResponse:
    return LLMResponse(content=[TextContent(text=text)], stop_reason=StopReason.END_TURN)


def _agent(agent_id: str) -> AgentDefinition:
    return AgentDefinition(
        agent_id=agent_id,
        name=agent_id,
        system_prompt=f"You are {agent_id}.",
    )


def _context(
    workflow_definition: WorkflowDefinition,
    agent_registry: AgentRegistry,
    memory_store: InMemoryStore | None = None,
) -> ExecutionContext:
    return ExecutionContext(
        run_id=_RUN_ID,
        workflow_definition=workflow_definition,
        shared_state=None,
        workflow_registry=None,
        agent_registry=agent_registry,
        tool_registry=ToolRegistry(),
        memory_store=memory_store or InMemoryStore(),
        policy_engine=PolicyEngine(),
        observer=_CapturingObserver(),
    )


# ---------------------------------------------------------------------------
# TestParallelSpecialistExecutor
# ---------------------------------------------------------------------------


class TestParallelSpecialistExecutor:
    async def test_execute_returns_aggregated_output(self):
        registry = AgentRegistry()
        registry.register(_agent("spec-a"))
        registry.register(_agent("spec-b"))

        llm = MockLLMProvider([_response("output A"), _response("output B")])
        wf = WorkflowDefinition(
            workflow_id="wf-1",
            name="test",
            pattern=PatternType.PARALLEL_SPECIALIST,
            agent_ids=["spec-a", "spec-b"],
        )
        result = await ParallelSpecialistExecutor(llm).execute(_context(wf, registry), "go")

        assert isinstance(result, WorkflowResult)
        assert "output A" in result.output
        assert "output B" in result.output
        assert result.run_id == _RUN_ID
        assert result.workflow_id == "wf-1"

    async def test_execute_all_agents_are_called(self):
        registry = AgentRegistry()
        registry.register(_agent("spec-a"))
        registry.register(_agent("spec-b"))

        llm = MockLLMProvider([_response("A"), _response("B")])
        wf = WorkflowDefinition(
            workflow_id="wf-1",
            name="test",
            pattern=PatternType.PARALLEL_SPECIALIST,
            agent_ids=["spec-a", "spec-b"],
        )
        result = await ParallelSpecialistExecutor(llm).execute(_context(wf, registry), "go")

        assert len(result.agent_results) == 2
        ids = {r.agent_id for r in result.agent_results}
        assert ids == {"spec-a", "spec-b"}

    async def test_execute_single_agent(self):
        registry = AgentRegistry()
        registry.register(_agent("solo"))

        llm = MockLLMProvider([_response("solo result")])
        wf = WorkflowDefinition(
            workflow_id="wf-1",
            name="test",
            pattern=PatternType.PARALLEL_SPECIALIST,
            agent_ids=["solo"],
        )
        result = await ParallelSpecialistExecutor(llm).execute(_context(wf, registry), "go")

        assert result.output == "## [solo]\n\nsolo result"
        assert len(result.agent_results) == 1

    async def test_execute_empty_agent_ids_returns_empty_output(self):
        wf = WorkflowDefinition(
            workflow_id="wf-1",
            name="test",
            pattern=PatternType.PARALLEL_SPECIALIST,
            agent_ids=[],
        )
        result = await ParallelSpecialistExecutor(
            MockLLMProvider([])
        ).execute(_context(wf, AgentRegistry()), "go")

        assert result.output == ""
        assert result.agent_results == []

    async def test_execute_agent_failure_raises_pattern_execution_error(self):
        registry = AgentRegistry()
        registry.register(_agent("failing-agent"))

        # Empty queue → RuntimeError on complete()
        llm = MockLLMProvider([])
        wf = WorkflowDefinition(
            workflow_id="wf-1",
            name="test",
            pattern=PatternType.PARALLEL_SPECIALIST,
            agent_ids=["failing-agent"],
        )
        with pytest.raises(PatternExecutionError, match="failing-agent"):
            await ParallelSpecialistExecutor(llm).execute(_context(wf, registry), "go")

    async def test_execute_with_reviewer_agent(self):
        registry = AgentRegistry()
        registry.register(_agent("spec-a"))
        registry.register(_agent("spec-b"))
        registry.register(_agent("reviewer"))

        llm = MockLLMProvider([
            _response("output A"),
            _response("output B"),
            _response("reviewed final"),
        ])
        wf = WorkflowDefinition(
            workflow_id="wf-1",
            name="test",
            pattern=PatternType.PARALLEL_SPECIALIST,
            agent_ids=["spec-a", "spec-b"],
            pattern_config={"reviewer_agent_id": "reviewer"},
        )
        result = await ParallelSpecialistExecutor(llm).execute(_context(wf, registry), "go")

        assert result.output == "reviewed final"
        assert len(result.agent_results) == 3
        assert result.agent_results[2].agent_id == "reviewer"

    async def test_execute_reviewer_not_called_without_config(self):
        registry = AgentRegistry()
        registry.register(_agent("spec-a"))

        llm = MockLLMProvider([_response("just A")])
        wf = WorkflowDefinition(
            workflow_id="wf-1",
            name="test",
            pattern=PatternType.PARALLEL_SPECIALIST,
            agent_ids=["spec-a"],
        )
        result = await ParallelSpecialistExecutor(llm).execute(_context(wf, registry), "go")

        assert len(result.agent_results) == 1
        assert not llm._queue  # queue fully consumed (no extra reviewer call)


# ---------------------------------------------------------------------------
# TestRouterExecutor
# ---------------------------------------------------------------------------


class TestRouterExecutor:
    async def test_execute_dispatches_to_correct_agent(self):
        registry = AgentRegistry()
        registry.register(_agent("classifier"))
        registry.register(_agent("billing-agent"))
        registry.register(_agent("support-agent"))

        llm = MockLLMProvider([
            _response("billing"),        # classifier output
            _response("billing result"), # billing-agent output
        ])
        wf = WorkflowDefinition(
            workflow_id="wf-1",
            name="test",
            pattern=PatternType.ROUTER,
            pattern_config={
                "classifier_agent_id": "classifier",
                "routes": {"billing": "billing-agent", "support": "support-agent"},
            },
        )
        result = await RouterExecutor(llm).execute(_context(wf, registry), "I have a billing issue")

        assert result.output == "billing result"
        assert result.workflow_id == "wf-1"

    async def test_execute_route_label_stripped_and_lowercased(self):
        registry = AgentRegistry()
        registry.register(_agent("classifier"))
        registry.register(_agent("billing-agent"))

        llm = MockLLMProvider([
            _response("  Billing\n"),    # classifier with whitespace and mixed case
            _response("billing result"),
        ])
        wf = WorkflowDefinition(
            workflow_id="wf-1",
            name="test",
            pattern=PatternType.ROUTER,
            pattern_config={
                "classifier_agent_id": "classifier",
                "routes": {"billing": "billing-agent"},
            },
        )
        result = await RouterExecutor(llm).execute(_context(wf, registry), "help")

        assert result.output == "billing result"

    async def test_execute_unknown_route_raises_pattern_execution_error(self):
        registry = AgentRegistry()
        registry.register(_agent("classifier"))

        llm = MockLLMProvider([_response("unknown-route")])
        wf = WorkflowDefinition(
            workflow_id="wf-1",
            name="test",
            pattern=PatternType.ROUTER,
            pattern_config={
                "classifier_agent_id": "classifier",
                "routes": {"billing": "billing-agent"},
            },
        )
        with pytest.raises(PatternExecutionError, match="unknown-route"):
            await RouterExecutor(llm).execute(_context(wf, registry), "help")

    async def test_execute_target_agent_not_in_registry_raises_agent_not_found(self):
        registry = AgentRegistry()
        registry.register(_agent("classifier"))
        # billing-agent deliberately NOT registered

        llm = MockLLMProvider([_response("billing")])
        wf = WorkflowDefinition(
            workflow_id="wf-1",
            name="test",
            pattern=PatternType.ROUTER,
            pattern_config={
                "classifier_agent_id": "classifier",
                "routes": {"billing": "billing-agent"},
            },
        )
        with pytest.raises(AgentNotFound):
            await RouterExecutor(llm).execute(_context(wf, registry), "help")

    async def test_execute_passes_original_input_to_target(self):
        registry = AgentRegistry()
        registry.register(_agent("classifier"))
        registry.register(_agent("target-agent"))

        llm = MockLLMProvider([_response("billing"), _response("target output")])
        wf = WorkflowDefinition(
            workflow_id="wf-1",
            name="test",
            pattern=PatternType.ROUTER,
            pattern_config={
                "classifier_agent_id": "classifier",
                "routes": {"billing": "target-agent"},
            },
        )
        memory = InMemoryStore()
        context = _context(wf, registry, memory_store=memory)
        await RouterExecutor(llm).execute(context, "original workflow question")

        # Target agent's history must contain the original workflow input, not the route label
        target_history = memory.get_history(_RUN_ID, "target-agent")
        user_inputs = [
            m.content for m in target_history if m.role == Role.USER and isinstance(m.content, str)
        ]
        assert "original workflow question" in user_inputs

    async def test_execute_agent_results_includes_classifier_and_target(self):
        registry = AgentRegistry()
        registry.register(_agent("classifier"))
        registry.register(_agent("target-agent"))

        llm = MockLLMProvider([_response("billing"), _response("target output")])
        wf = WorkflowDefinition(
            workflow_id="wf-1",
            name="test",
            pattern=PatternType.ROUTER,
            pattern_config={
                "classifier_agent_id": "classifier",
                "routes": {"billing": "target-agent"},
            },
        )
        result = await RouterExecutor(llm).execute(_context(wf, registry), "help")

        assert len(result.agent_results) == 2
        assert result.agent_results[0].agent_id == "classifier"
        assert result.agent_results[1].agent_id == "target-agent"


# ---------------------------------------------------------------------------
# TestPlannerExecutorObserverExecutor — placeholder (Phase 6B)
# ---------------------------------------------------------------------------


class TestPlannerExecutorObserverExecutor:
    # TODO: test execute() runs planner → executor → observer sequence
    # TODO: test execute() exits loop when observer signals DONE
    # TODO: test execute() respects max_iterations from pattern_config
    # TODO: test execute() stores plan in shared_state
    pass

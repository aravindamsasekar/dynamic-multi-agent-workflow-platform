"""Unit tests for Orchestrator."""

from __future__ import annotations

import pytest

from platform.core.exceptions import PatternExecutionError, WorkflowNotFound
from platform.core.interfaces.observer import IObserver
from platform.core.models.agent import AgentDefinition
from platform.core.models.events import EventType, WorkflowEvent
from platform.core.models.message import LLMResponse, StopReason, TextContent
from platform.core.models.workflow import PatternType, RunStatus, WorkflowDefinition, WorkflowResult
from platform.llm.mock_provider import MockLLMProvider
from platform.memory.in_memory_store import InMemoryStore
from platform.orchestrator.orchestrator import Orchestrator
from platform.orchestrator.run_manager import RunManager
from platform.policy.engine import PolicyEngine
from platform.registries.agent_registry import AgentRegistry
from platform.registries.tool_registry import ToolRegistry
from platform.registries.workflow_registry import WorkflowRegistry
from platform.state.shared_state import SharedState


class _CapturingObserver(IObserver):
    def __init__(self) -> None:
        self.events: list[WorkflowEvent] = []

    def on_event(self, event: WorkflowEvent) -> None:
        self.events.append(event)


def _text(text: str) -> LLMResponse:
    return LLMResponse(content=[TextContent(text=text)], stop_reason=StopReason.END_TURN)


def _make_orchestrator(
    llm_provider: MockLLMProvider,
    observer: IObserver | None = None,
) -> tuple[Orchestrator, RunManager, _CapturingObserver]:
    wf_registry = WorkflowRegistry()
    ag_registry = AgentRegistry()

    wf_registry.register(
        WorkflowDefinition(
            workflow_id="test_wf",
            name="Test Workflow",
            pattern=PatternType.PARALLEL_SPECIALIST,
            agent_ids=["agent_a"],
            pattern_config={"strategy": "concatenate"},
        )
    )
    ag_registry.register(
        AgentDefinition(
            agent_id="agent_a",
            name="Agent A",
            system_prompt="You are helpful.",
            tool_names=[],
        )
    )

    obs = observer or _CapturingObserver()
    run_manager = RunManager()
    orch = Orchestrator(
        workflow_registry=wf_registry,
        agent_registry=ag_registry,
        tool_registry=ToolRegistry(),
        memory_store=InMemoryStore(),
        policy_engine=PolicyEngine(),
        observer=obs,
        run_manager=run_manager,
        llm_provider=llm_provider,
        shared_state=SharedState(),
    )
    return orch, run_manager, obs  # type: ignore[return-value]


class TestOrchestrator:
    async def test_run_returns_workflow_result(self) -> None:
        llm = MockLLMProvider([_text("final answer")])
        orch, _, _ = _make_orchestrator(llm)
        result = await orch.run("test_wf", "test input")
        assert isinstance(result, WorkflowResult)
        assert result.workflow_id == "test_wf"
        assert "final answer" in result.output

    async def test_run_creates_completed_run(self) -> None:
        llm = MockLLMProvider([_text("done")])
        orch, run_manager, _ = _make_orchestrator(llm)
        result = await orch.run("test_wf", "hello")
        run = run_manager.get_run(result.run_id)
        assert run.status == RunStatus.COMPLETED
        assert "done" in run.output

    async def test_run_emits_started_and_completed_events(self) -> None:
        llm = MockLLMProvider([_text("result")])
        observer = _CapturingObserver()
        orch, _, _ = _make_orchestrator(llm, observer)
        await orch.run("test_wf", "input")
        event_types = [e.event_type for e in observer.events]
        assert EventType.WORKFLOW_STARTED in event_types
        assert EventType.WORKFLOW_COMPLETED in event_types

    async def test_run_marks_failed_on_llm_error(self) -> None:
        llm = MockLLMProvider([])  # immediately exhausted
        orch, run_manager, _ = _make_orchestrator(llm)
        with pytest.raises(PatternExecutionError):
            await orch.run("test_wf", "input")
        failed = [r for r in run_manager.list_runs() if r.status == RunStatus.FAILED]
        assert len(failed) == 1

    async def test_run_raises_workflow_not_found(self) -> None:
        llm = MockLLMProvider([])
        orch, _, _ = _make_orchestrator(llm)
        with pytest.raises(WorkflowNotFound):
            await orch.run("nonexistent_wf", "input")

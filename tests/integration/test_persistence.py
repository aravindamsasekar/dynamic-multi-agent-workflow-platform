"""Integration tests: execute a workflow and verify all DB tables are populated."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from platform.config.loader import ConfigLoader
from platform.core.models.message import LLMResponse, StopReason, TextContent, ToolUseContent
from platform.core.models.workflow import RunStatus
from platform.llm.mock_provider import MockLLMProvider
from platform.memory.in_memory_store import InMemoryStore
from platform.observability.composite_observer import CompositeObserver
from platform.observability.console_observer import ConsoleObserver
from platform.observability.persisting_observer import PersistingObserver
from platform.orchestrator.orchestrator import Orchestrator
from platform.orchestrator.run_manager import RunManager
from platform.persistence.database import Base
from platform.persistence.repositories.agent_repo import AgentRepository
from platform.persistence.repositories.event_repo import EventRepository
from platform.persistence.repositories.run_repo import RunRepository
from platform.persistence.repositories.tool_repo import ToolRepository
from platform.policy.engine import PolicyEngine
from platform.registries.agent_registry import AgentRegistry
from platform.registries.tool_registry import ToolRegistry
from platform.registries.workflow_registry import WorkflowRegistry
from platform.state.shared_state import SharedState

_WORKFLOWS_DIR = Path(__file__).resolve().parent.parent.parent / "workflows"


def _text(text: str) -> LLMResponse:
    return LLMResponse(content=[TextContent(text=text)], stop_reason=StopReason.END_TURN)


def _tool_use(tool_id: str, tool_name: str, input: dict) -> LLMResponse:
    return LLMResponse(
        content=[ToolUseContent(id=tool_id, name=tool_name, input=input)],
        stop_reason=StopReason.TOOL_USE,
    )


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


@pytest.fixture
def session(session_factory):
    with session_factory() as s:
        yield s


def _build_orchestrator(session_factory, llm):
    wf_reg = WorkflowRegistry()
    ag_reg = AgentRegistry()
    tl_reg = ToolRegistry()
    ConfigLoader(wf_reg, ag_reg, tl_reg).load_one(_WORKFLOWS_DIR / "incident_commander")

    observer = CompositeObserver([
        ConsoleObserver(),
        PersistingObserver(session_factory),
    ])

    run_manager = RunManager()
    return (
        Orchestrator(
            workflow_registry=wf_reg,
            agent_registry=ag_reg,
            tool_registry=tl_reg,
            memory_store=InMemoryStore(),
            policy_engine=PolicyEngine(),
            observer=observer,
            run_manager=run_manager,
            llm_provider=llm,
            shared_state=SharedState(),
        ),
        run_manager,
    )


class TestWorkflowPersistenceIntegration:
    async def test_run_creates_workflow_run_row(self, session_factory, session) -> None:
        llm = MockLLMProvider([
            _text("Metrics: CPU 94%."),
            _text("Logs: DB timeouts."),
            _text("Deploy: v2.4.1."),
            _text("Root cause: DB pool exhaustion."),
        ])
        orch, _ = _build_orchestrator(session_factory, llm)
        result = await orch.run("incident_commander", "Production alert")

        row = RunRepository().get(session, result.run_id)
        assert row is not None
        assert row.workflow_id == "incident_commander"
        assert row.status == "completed"
        assert row.input == "Production alert"
        assert row.output == "Root cause: DB pool exhaustion."

    async def test_run_persists_agent_results(self, session_factory, session) -> None:
        llm = MockLLMProvider([
            _text("Metrics: CPU 94%."),
            _text("Logs: DB timeouts."),
            _text("Deploy: v2.4.1."),
            _text("Root cause: DB pool exhaustion."),
        ])
        orch, _ = _build_orchestrator(session_factory, llm)
        result = await orch.run("incident_commander", "Production alert")

        rows = AgentRepository().list_for_run(session, result.run_id)
        assert len(rows) == 4
        agent_ids = {r.agent_id for r in rows}
        assert "metrics_agent" in agent_ids
        assert "logs_agent" in agent_ids
        assert "deployment_agent" in agent_ids
        assert "reviewer_agent" in agent_ids

    async def test_run_persists_tool_calls(self, session_factory, session) -> None:
        llm = MockLLMProvider([
            _tool_use("tc-1", "mock_metrics_tool", {"time_range": "last_5m"}),
            _text("CPU at 94% — critical."),
            _text("Logs: DB timeouts."),
            _text("Deploy: v2.4.1."),
            _text("Root cause: DB pool exhaustion."),
        ])
        orch, _ = _build_orchestrator(session_factory, llm)
        result = await orch.run("incident_commander", "Production alert")

        rows = ToolRepository().list_for_run(session, result.run_id)
        assert len(rows) == 1
        assert rows[0].tool_name == "mock_metrics_tool"
        assert rows[0].is_error is False
        assert '"time_range"' in rows[0].input

    async def test_run_persists_events(self, session_factory, session) -> None:
        llm = MockLLMProvider([
            _text("Metrics."),
            _text("Logs."),
            _text("Deploy."),
            _text("Summary."),
        ])
        orch, _ = _build_orchestrator(session_factory, llm)
        result = await orch.run("incident_commander", "Production alert")

        rows = EventRepository().list_for_run(session, result.run_id)
        event_types = [r.event_type for r in rows]
        assert "workflow_started" in event_types
        assert "workflow_completed" in event_types
        assert "agent_completed" in event_types

    async def test_failed_run_persisted(self, session_factory, session) -> None:
        from platform.llm.mock_provider import MockLLMProvider as MLP

        class _FailingLLM:
            async def complete(self, messages, tools=None):
                raise RuntimeError("LLM exploded")

        wf_reg = WorkflowRegistry()
        ag_reg = AgentRegistry()
        tl_reg = ToolRegistry()
        ConfigLoader(wf_reg, ag_reg, tl_reg).load_one(_WORKFLOWS_DIR / "incident_commander")

        observer = CompositeObserver([
            PersistingObserver(session_factory),
        ])
        run_manager = RunManager()
        orch = Orchestrator(
            workflow_registry=wf_reg,
            agent_registry=ag_reg,
            tool_registry=tl_reg,
            memory_store=InMemoryStore(),
            policy_engine=PolicyEngine(),
            observer=observer,
            run_manager=run_manager,
            llm_provider=_FailingLLM(),
            shared_state=SharedState(),
        )

        with pytest.raises(Exception, match="LLM exploded"):
            await orch.run("incident_commander", "trigger failure")

        all_runs = RunRepository().list_all(session)
        assert len(all_runs) == 1
        assert all_runs[0].status == "failed"
        assert all_runs[0].error is not None
        assert len(all_runs[0].error) > 0

    async def test_event_payload_contains_input(self, session_factory, session) -> None:
        llm = MockLLMProvider([
            _text("Metrics."),
            _text("Logs."),
            _text("Deploy."),
            _text("Summary."),
        ])
        orch, _ = _build_orchestrator(session_factory, llm)
        result = await orch.run("incident_commander", "specific input text")

        events = EventRepository().list_for_run(session, result.run_id)
        started = next(e for e in events if e.event_type == "workflow_started")
        assert "specific input text" in started.payload

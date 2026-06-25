"""Tests for PersistingObserver and CompositeObserver."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from platform.core.models.events import (
    AgentCompletedEvent,
    EventType,
    ToolCompletedEvent,
    WorkflowCompletedEvent,
    WorkflowFailedEvent,
    WorkflowStartedEvent,
    WorkflowEvent,
)
from platform.observability.composite_observer import CompositeObserver
from platform.observability.persisting_observer import PersistingObserver
from platform.persistence.database import Base
from platform.persistence.repositories.agent_repo import AgentRepository
from platform.persistence.repositories.event_repo import EventRepository
from platform.persistence.repositories.run_repo import RunRepository
from platform.persistence.repositories.tool_repo import ToolRepository

_TS = datetime(2024, 6, 1, 10, 0, 0)


@pytest.fixture
def db_session_factory():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


@pytest.fixture
def observer(db_session_factory):
    return PersistingObserver(db_session_factory)


@pytest.fixture
def session(db_session_factory):
    with db_session_factory() as s:
        yield s


# ---------------------------------------------------------------------------
# PersistingObserver
# ---------------------------------------------------------------------------


class TestPersistingObserver:
    def test_workflow_started_creates_run_row(self, observer, session) -> None:
        observer.on_event(
            WorkflowStartedEvent(
                run_id="run-ps-1",
                timestamp=_TS,
                data={"workflow_id": "wf-x", "input": "my input"},
            )
        )
        row = RunRepository().get(session, "run-ps-1")
        assert row is not None
        assert row.workflow_id == "wf-x"
        assert row.input == "my input"
        assert row.status == "running"

    def test_workflow_started_inserts_event_row(self, observer, session) -> None:
        observer.on_event(
            WorkflowStartedEvent(
                run_id="run-ps-2",
                timestamp=_TS,
                data={"workflow_id": "wf-x", "input": "hi"},
            )
        )
        rows = EventRepository().list_for_run(session, "run-ps-2")
        assert len(rows) == 1
        assert rows[0].event_type == EventType.WORKFLOW_STARTED.value

    def test_workflow_completed_updates_run(self, observer, session) -> None:
        observer.on_event(
            WorkflowStartedEvent(
                run_id="run-ps-3",
                timestamp=_TS,
                data={"workflow_id": "wf-x", "input": "q"},
            )
        )
        observer.on_event(
            WorkflowCompletedEvent(
                run_id="run-ps-3",
                timestamp=_TS,
                data={"output": "the answer"},
            )
        )
        row = RunRepository().get(session, "run-ps-3")
        assert row.status == "completed"
        assert row.output == "the answer"

    def test_workflow_failed_updates_run(self, observer, session) -> None:
        observer.on_event(
            WorkflowStartedEvent(
                run_id="run-ps-4",
                timestamp=_TS,
                data={"workflow_id": "wf-x", "input": "q"},
            )
        )
        observer.on_event(
            WorkflowFailedEvent(
                run_id="run-ps-4",
                timestamp=_TS,
                data={"error": "something broke"},
            )
        )
        row = RunRepository().get(session, "run-ps-4")
        assert row.status == "failed"
        assert row.error == "something broke"

    def test_agent_completed_inserts_agent_result(self, observer, session) -> None:
        observer.on_event(
            WorkflowStartedEvent(
                run_id="run-ps-5",
                timestamp=_TS,
                data={"workflow_id": "wf-x", "input": "q"},
            )
        )
        observer.on_event(
            AgentCompletedEvent(
                run_id="run-ps-5",
                timestamp=_TS,
                data={"agent_id": "my-agent", "output": "agent output"},
            )
        )
        rows = AgentRepository().list_for_run(session, "run-ps-5")
        assert len(rows) == 1
        assert rows[0].agent_id == "my-agent"
        assert rows[0].output == "agent output"

    def test_tool_completed_inserts_tool_call(self, observer, session) -> None:
        observer.on_event(
            WorkflowStartedEvent(
                run_id="run-ps-6",
                timestamp=_TS,
                data={"workflow_id": "wf-x", "input": "q"},
            )
        )
        observer.on_event(
            ToolCompletedEvent(
                run_id="run-ps-6",
                timestamp=_TS,
                data={
                    "tool_name": "search",
                    "tool_input": {"query": "hello"},
                    "result": "found it",
                    "is_error": False,
                },
            )
        )
        rows = ToolRepository().list_for_run(session, "run-ps-6")
        assert len(rows) == 1
        assert rows[0].tool_name == "search"
        assert rows[0].output == "found it"
        assert rows[0].is_error is False
        assert '"query"' in rows[0].input

    def test_tool_completed_with_error_flag(self, observer, session) -> None:
        observer.on_event(
            WorkflowStartedEvent(
                run_id="run-ps-7",
                timestamp=_TS,
                data={"workflow_id": "wf-x", "input": "q"},
            )
        )
        observer.on_event(
            ToolCompletedEvent(
                run_id="run-ps-7",
                timestamp=_TS,
                data={
                    "tool_name": "bad_tool",
                    "tool_input": {},
                    "result": "error text",
                    "is_error": True,
                },
            )
        )
        rows = ToolRepository().list_for_run(session, "run-ps-7")
        assert rows[0].is_error is True

    def test_all_event_types_insert_event_row(self, observer, session) -> None:
        observer.on_event(
            WorkflowStartedEvent(
                run_id="run-ps-8",
                timestamp=_TS,
                data={"workflow_id": "wf", "input": "x"},
            )
        )
        observer.on_event(
            AgentCompletedEvent(
                run_id="run-ps-8",
                timestamp=_TS,
                data={"agent_id": "a", "output": "o"},
            )
        )
        observer.on_event(
            WorkflowCompletedEvent(
                run_id="run-ps-8",
                timestamp=_TS,
                data={"output": "done"},
            )
        )
        rows = EventRepository().list_for_run(session, "run-ps-8")
        assert len(rows) == 3


# ---------------------------------------------------------------------------
# CompositeObserver
# ---------------------------------------------------------------------------


class TestCompositeObserver:
    def test_calls_all_observers(self) -> None:
        obs_a = MagicMock()
        obs_b = MagicMock()
        composite = CompositeObserver([obs_a, obs_b])

        event = WorkflowStartedEvent(run_id="r", data={"workflow_id": "wf", "input": "x"})
        composite.on_event(event)

        obs_a.on_event.assert_called_once_with(event)
        obs_b.on_event.assert_called_once_with(event)

    def test_calls_observers_in_order(self) -> None:
        call_order: list[str] = []

        class _Obs:
            def __init__(self, name: str) -> None:
                self._name = name

            def on_event(self, event) -> None:
                call_order.append(self._name)

        composite = CompositeObserver([_Obs("first"), _Obs("second"), _Obs("third")])
        composite.on_event(WorkflowStartedEvent(run_id="r", data={"workflow_id": "w", "input": "x"}))
        assert call_order == ["first", "second", "third"]

    def test_empty_observers_list(self) -> None:
        composite = CompositeObserver([])
        composite.on_event(WorkflowStartedEvent(run_id="r", data={"workflow_id": "w", "input": "x"}))

    def test_persisting_and_console_combined(self, db_session_factory) -> None:
        from platform.observability.console_observer import ConsoleObserver
        import io, contextlib

        buf = io.StringIO()
        console = ConsoleObserver()
        persisting = PersistingObserver(db_session_factory)
        composite = CompositeObserver([console, persisting])

        event = WorkflowStartedEvent(
            run_id="run-combo",
            timestamp=_TS,
            data={"workflow_id": "wf", "input": "x"},
        )
        with contextlib.redirect_stdout(buf):
            composite.on_event(event)

        assert "run-combo" in buf.getvalue()

        with db_session_factory() as s:
            row = RunRepository().get(s, "run-combo")
        assert row is not None

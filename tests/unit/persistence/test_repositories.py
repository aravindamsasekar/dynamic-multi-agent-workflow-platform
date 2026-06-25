"""Repository unit tests — all run against SQLite in-memory."""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from platform.persistence.database import Base
from platform.persistence.models import AgentResultRow, EventRow, ToolCallRow, WorkflowRunRow
from platform.persistence.repositories.agent_repo import AgentRepository
from platform.persistence.repositories.event_repo import EventRepository
from platform.persistence.repositories.run_repo import RunRepository
from platform.persistence.repositories.tool_repo import ToolRepository

_TS = datetime(2024, 1, 1, 12, 0, 0)
_TS2 = datetime(2024, 1, 1, 12, 0, 1)


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as s:
        yield s


@pytest.fixture
def run_with_session(session):
    """Pre-insert a workflow_run row and return (session, run_id)."""
    repo = RunRepository()
    repo.create(
        session,
        run_id="run-001",
        workflow_id="wf-a",
        status="running",
        input="hello",
        created_at=_TS,
        updated_at=_TS,
    )
    session.commit()
    return session, "run-001"


# ---------------------------------------------------------------------------
# RunRepository
# ---------------------------------------------------------------------------


class TestRunRepository:
    def test_create_and_get(self, session) -> None:
        repo = RunRepository()
        repo.create(
            session,
            run_id="run-1",
            workflow_id="wf-x",
            status="running",
            input="test input",
            created_at=_TS,
            updated_at=_TS,
        )
        session.commit()

        row = repo.get(session, "run-1")
        assert row is not None
        assert row.run_id == "run-1"
        assert row.workflow_id == "wf-x"
        assert row.status == "running"
        assert row.input == "test input"
        assert row.output is None
        assert row.error is None

    def test_get_missing_returns_none(self, session) -> None:
        repo = RunRepository()
        assert repo.get(session, "does-not-exist") is None

    def test_list_all_empty(self, session) -> None:
        repo = RunRepository()
        assert repo.list_all(session) == []

    def test_list_all_returns_rows_newest_first(self, session) -> None:
        repo = RunRepository()
        ts_early = datetime(2024, 1, 1, 10, 0, 0)
        ts_late = datetime(2024, 1, 1, 11, 0, 0)
        repo.create(session, "run-a", "wf", "running", "a", created_at=ts_early, updated_at=ts_early)
        repo.create(session, "run-b", "wf", "running", "b", created_at=ts_late, updated_at=ts_late)
        session.commit()

        rows = repo.list_all(session)
        assert [r.run_id for r in rows] == ["run-b", "run-a"]

    def test_update_status_to_completed(self, run_with_session) -> None:
        session, run_id = run_with_session
        repo = RunRepository()
        repo.update_status(session, run_id, "completed", updated_at=_TS2, output="done")
        session.commit()

        row = repo.get(session, run_id)
        assert row.status == "completed"
        assert row.output == "done"
        assert row.updated_at == _TS2

    def test_update_status_to_failed(self, run_with_session) -> None:
        session, run_id = run_with_session
        repo = RunRepository()
        repo.update_status(session, run_id, "failed", updated_at=_TS2, error="boom")
        session.commit()

        row = repo.get(session, run_id)
        assert row.status == "failed"
        assert row.error == "boom"

    def test_update_status_missing_run_is_noop(self, session) -> None:
        repo = RunRepository()
        repo.update_status(session, "ghost-run", "completed", updated_at=_TS2)
        session.commit()


# ---------------------------------------------------------------------------
# AgentRepository
# ---------------------------------------------------------------------------


class TestAgentRepository:
    def test_create_and_list(self, run_with_session) -> None:
        session, run_id = run_with_session
        repo = AgentRepository()
        repo.create(session, run_id=run_id, agent_id="agent-1", output="result", created_at=_TS)
        session.commit()

        rows = repo.list_for_run(session, run_id)
        assert len(rows) == 1
        assert rows[0].agent_id == "agent-1"
        assert rows[0].output == "result"

    def test_list_sorted_by_created_at_asc(self, run_with_session) -> None:
        session, run_id = run_with_session
        repo = AgentRepository()
        ts_a = datetime(2024, 1, 1, 12, 0, 5)
        ts_b = datetime(2024, 1, 1, 12, 0, 1)
        repo.create(session, run_id, "agent-a", "out-a", ts_a)
        repo.create(session, run_id, "agent-b", "out-b", ts_b)
        session.commit()

        rows = repo.list_for_run(session, run_id)
        assert [r.agent_id for r in rows] == ["agent-b", "agent-a"]

    def test_list_for_run_empty(self, run_with_session) -> None:
        session, run_id = run_with_session
        repo = AgentRepository()
        assert repo.list_for_run(session, run_id) == []

    def test_list_does_not_cross_runs(self, session) -> None:
        run_repo = RunRepository()
        run_repo.create(session, "r1", "wf", "running", "i", _TS, _TS)
        run_repo.create(session, "r2", "wf", "running", "i", _TS, _TS)
        session.commit()

        repo = AgentRepository()
        repo.create(session, "r1", "agent-x", "out", _TS)
        session.commit()

        assert repo.list_for_run(session, "r2") == []


# ---------------------------------------------------------------------------
# ToolRepository
# ---------------------------------------------------------------------------


class TestToolRepository:
    def test_create_with_dict_input(self, run_with_session) -> None:
        session, run_id = run_with_session
        repo = ToolRepository()
        repo.create(session, run_id, "my_tool", {"key": "value"}, "ok", False, _TS)
        session.commit()

        rows = repo.list_for_run(session, run_id)
        assert len(rows) == 1
        assert rows[0].tool_name == "my_tool"
        assert rows[0].input == '{"key": "value"}'
        assert rows[0].output == "ok"
        assert rows[0].is_error is False

    def test_create_with_error(self, run_with_session) -> None:
        session, run_id = run_with_session
        repo = ToolRepository()
        repo.create(session, run_id, "bad_tool", None, "error msg", True, _TS)
        session.commit()

        rows = repo.list_for_run(session, run_id)
        assert rows[0].is_error is True
        assert rows[0].input is None

    def test_list_sorted_asc(self, run_with_session) -> None:
        session, run_id = run_with_session
        repo = ToolRepository()
        ts_a = datetime(2024, 1, 1, 12, 0, 10)
        ts_b = datetime(2024, 1, 1, 12, 0, 2)
        repo.create(session, run_id, "tool-a", None, None, False, ts_a)
        repo.create(session, run_id, "tool-b", None, None, False, ts_b)
        session.commit()

        rows = repo.list_for_run(session, run_id)
        assert [r.tool_name for r in rows] == ["tool-b", "tool-a"]


# ---------------------------------------------------------------------------
# EventRepository
# ---------------------------------------------------------------------------


class TestEventRepository:
    def test_create_and_list(self, run_with_session) -> None:
        session, run_id = run_with_session
        repo = EventRepository()
        repo.create(session, run_id, "workflow_started", {"workflow_id": "wf-a"}, _TS)
        session.commit()

        rows = repo.list_for_run(session, run_id)
        assert len(rows) == 1
        assert rows[0].event_type == "workflow_started"
        assert '"workflow_id"' in rows[0].payload

    def test_none_payload_stored_as_none(self, run_with_session) -> None:
        session, run_id = run_with_session
        repo = EventRepository()
        repo.create(session, run_id, "some_event", None, _TS)
        session.commit()

        rows = repo.list_for_run(session, run_id)
        assert rows[0].payload is None

    def test_list_sorted_asc(self, run_with_session) -> None:
        session, run_id = run_with_session
        repo = EventRepository()
        ts_a = datetime(2024, 1, 1, 12, 0, 20)
        ts_b = datetime(2024, 1, 1, 12, 0, 3)
        repo.create(session, run_id, "event-a", None, ts_a)
        repo.create(session, run_id, "event-b", None, ts_b)
        session.commit()

        rows = repo.list_for_run(session, run_id)
        assert [r.event_type for r in rows] == ["event-b", "event-a"]

    def test_multiple_event_types(self, run_with_session) -> None:
        session, run_id = run_with_session
        repo = EventRepository()
        for i, et in enumerate(["workflow_started", "agent_called", "workflow_completed"]):
            ts = datetime(2024, 1, 1, 12, 0, i)
            repo.create(session, run_id, et, None, ts)
        session.commit()

        rows = repo.list_for_run(session, run_id)
        assert len(rows) == 3
        assert [r.event_type for r in rows] == [
            "workflow_started", "agent_called", "workflow_completed"
        ]

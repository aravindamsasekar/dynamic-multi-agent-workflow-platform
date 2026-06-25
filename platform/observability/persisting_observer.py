"""PersistingObserver — writes every workflow event immediately to SQLite."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import sessionmaker, Session

from platform.core.interfaces.observer import IObserver
from platform.core.models.events import EventType, WorkflowEvent
from platform.persistence.repositories.agent_repo import AgentRepository
from platform.persistence.repositories.event_repo import EventRepository
from platform.persistence.repositories.run_repo import RunRepository
from platform.persistence.repositories.tool_repo import ToolRepository

_run_repo = RunRepository()
_agent_repo = AgentRepository()
_tool_repo = ToolRepository()
_event_repo = EventRepository()


class PersistingObserver(IObserver):
    """Writes workflow events to the database immediately on receipt.

    No buffering — every event triggers a commit so partial state is always
    recoverable after a crash.
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def on_event(self, event: WorkflowEvent) -> None:
        with self._session_factory() as session:
            self._handle(session, event)
            session.commit()

    def _handle(self, session: Session, event: WorkflowEvent) -> None:
        now: datetime = event.timestamp

        if event.event_type == EventType.WORKFLOW_STARTED:
            _run_repo.create(
                session,
                run_id=event.run_id,
                workflow_id=event.data.get("workflow_id", ""),
                status="running",
                input=event.data.get("input", ""),
                created_at=now,
                updated_at=now,
            )

        elif event.event_type == EventType.WORKFLOW_COMPLETED:
            _run_repo.update_status(
                session,
                run_id=event.run_id,
                status="completed",
                updated_at=now,
                output=event.data.get("output"),
            )

        elif event.event_type == EventType.WORKFLOW_FAILED:
            _run_repo.update_status(
                session,
                run_id=event.run_id,
                status="failed",
                updated_at=now,
                error=event.data.get("error"),
            )

        elif event.event_type == EventType.AGENT_COMPLETED:
            _agent_repo.create(
                session,
                run_id=event.run_id,
                agent_id=event.data.get("agent_id", ""),
                output=event.data.get("output", ""),
                created_at=now,
            )

        elif event.event_type == EventType.TOOL_COMPLETED:
            _tool_repo.create(
                session,
                run_id=event.run_id,
                tool_name=event.data.get("tool_name", ""),
                input=event.data.get("tool_input"),
                output=event.data.get("result"),
                is_error=bool(event.data.get("is_error", False)),
                created_at=now,
            )

        _event_repo.create(
            session,
            run_id=event.run_id,
            event_type=event.event_type.value,
            payload=event.data if event.data else None,
            created_at=now,
        )

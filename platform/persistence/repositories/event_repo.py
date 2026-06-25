"""EventRepository — CRUD for events table."""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from platform.persistence.models import EventRow


class EventRepository:
    def create(
        self,
        session: Session,
        run_id: str,
        event_type: str,
        payload: dict | None,
        created_at: datetime,
    ) -> EventRow:
        payload_str = json.dumps(payload) if payload is not None else None
        row = EventRow(
            run_id=run_id,
            event_type=event_type,
            payload=payload_str,
            created_at=created_at,
        )
        session.add(row)
        return row

    def list_for_run(self, session: Session, run_id: str) -> list[EventRow]:
        stmt = (
            select(EventRow)
            .where(EventRow.run_id == run_id)
            .order_by(EventRow.created_at.asc())
        )
        return list(session.scalars(stmt).all())

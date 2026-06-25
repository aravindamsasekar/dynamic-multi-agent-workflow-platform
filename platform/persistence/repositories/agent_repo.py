"""AgentRepository — CRUD for agent_results table."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from platform.persistence.models import AgentResultRow


class AgentRepository:
    def create(
        self,
        session: Session,
        run_id: str,
        agent_id: str,
        output: str,
        created_at: datetime,
    ) -> AgentResultRow:
        row = AgentResultRow(
            run_id=run_id,
            agent_id=agent_id,
            output=output,
            created_at=created_at,
        )
        session.add(row)
        return row

    def list_for_run(self, session: Session, run_id: str) -> list[AgentResultRow]:
        stmt = (
            select(AgentResultRow)
            .where(AgentResultRow.run_id == run_id)
            .order_by(AgentResultRow.created_at.asc())
        )
        return list(session.scalars(stmt).all())

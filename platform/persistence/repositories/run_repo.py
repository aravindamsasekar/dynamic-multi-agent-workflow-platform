"""RunRepository — CRUD for workflow_runs table."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from platform.persistence.models import WorkflowRunRow


class RunRepository:
    def create(
        self,
        session: Session,
        run_id: str,
        workflow_id: str,
        status: str,
        input: str,
        created_at: datetime,
        updated_at: datetime,
    ) -> WorkflowRunRow:
        row = WorkflowRunRow(
            run_id=run_id,
            workflow_id=workflow_id,
            status=status,
            input=input,
            created_at=created_at,
            updated_at=updated_at,
        )
        session.add(row)
        return row

    def get(self, session: Session, run_id: str) -> WorkflowRunRow | None:
        return session.get(WorkflowRunRow, run_id)

    def list_all(self, session: Session) -> list[WorkflowRunRow]:
        stmt = select(WorkflowRunRow).order_by(WorkflowRunRow.created_at.desc())
        return list(session.scalars(stmt).all())

    def update_status(
        self,
        session: Session,
        run_id: str,
        status: str,
        updated_at: datetime,
        output: str | None = None,
        error: str | None = None,
    ) -> None:
        row = session.get(WorkflowRunRow, run_id)
        if row is None:
            return
        row.status = status
        row.updated_at = updated_at
        if output is not None:
            row.output = output
        if error is not None:
            row.error = error

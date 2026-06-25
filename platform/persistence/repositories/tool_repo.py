"""ToolRepository — CRUD for tool_calls table."""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from platform.persistence.models import ToolCallRow


class ToolRepository:
    def create(
        self,
        session: Session,
        run_id: str,
        tool_name: str,
        input: dict | str | None,
        output: str | None,
        is_error: bool,
        created_at: datetime,
    ) -> ToolCallRow:
        input_str = json.dumps(input) if isinstance(input, dict) else input
        row = ToolCallRow(
            run_id=run_id,
            tool_name=tool_name,
            input=input_str,
            output=output,
            is_error=is_error,
            created_at=created_at,
        )
        session.add(row)
        return row

    def list_for_run(self, session: Session, run_id: str) -> list[ToolCallRow]:
        stmt = (
            select(ToolCallRow)
            .where(ToolCallRow.run_id == run_id)
            .order_by(ToolCallRow.created_at.asc())
        )
        return list(session.scalars(stmt).all())

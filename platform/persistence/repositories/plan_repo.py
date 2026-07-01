"""PlanRepository — CRUD for generated_plans table."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from platform.persistence.models import GeneratedPlanRow
from platform.planner.models import GeneratedWorkflowPlan, ValidationResult
from platform.planner.serialization import plan_to_json, validation_to_json


class PlanRepository:
    def create(
        self,
        session: Session,
        plan: GeneratedWorkflowPlan,
        validation: ValidationResult,
        status: str = "pending_review",
    ) -> GeneratedPlanRow:
        now = datetime.utcnow()
        row = GeneratedPlanRow(
            plan_id=plan.plan_id,
            goal=plan.user_goal,
            status=status,
            plan_json=plan_to_json(plan),
            validation_json=validation_to_json(validation),
            execution_run_id=None,
            created_at=now,
            updated_at=now,
        )
        session.add(row)
        return row

    def get(self, session: Session, plan_id: str) -> GeneratedPlanRow | None:
        return session.get(GeneratedPlanRow, plan_id)

    def list_all(self, session: Session) -> list[GeneratedPlanRow]:
        stmt = select(GeneratedPlanRow).order_by(GeneratedPlanRow.created_at.desc())
        return list(session.scalars(stmt).all())

    def upgrade_preview_only_to_pending_review(self, session: Session) -> int:
        """Phase C startup migration: convert all preview_only rows to pending_review.

        Returns the number of rows updated. Called once at server startup so that
        plans generated before Phase C (when generated agents were not executable)
        become approvable. After the transition period this method can be removed.
        """
        rows = (
            session.query(GeneratedPlanRow)
            .filter(GeneratedPlanRow.status == "preview_only")
            .all()
        )
        now = datetime.utcnow()
        for row in rows:
            row.status = "pending_review"
            row.updated_at = now
        return len(rows)

    def update_status(
        self,
        session: Session,
        plan_id: str,
        status: str,
        execution_run_id: str | None = None,
    ) -> None:
        row = session.get(GeneratedPlanRow, plan_id)
        if row is None:
            return
        row.status = status
        row.updated_at = datetime.utcnow()
        if execution_run_id is not None:
            row.execution_run_id = execution_run_id

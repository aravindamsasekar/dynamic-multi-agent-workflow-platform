"""Unit tests for PlanRepository — SQLite in-memory."""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from platform.persistence.database import Base
from platform.persistence.repositories.plan_repo import PlanRepository
from platform.planner.models import (
    GeneratedWorkflowPlan,
    GoalAnalysis,
    GuardrailConfig,
    RiskLevel,
    TaskType,
    ValidationError,
    ValidationResult,
    ValidationWarning,
)

_TS = datetime(2025, 1, 1, 12, 0, 0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
def repo() -> PlanRepository:
    return PlanRepository()


def _make_plan(plan_id: str = "plan-001") -> GeneratedWorkflowPlan:
    return GeneratedWorkflowPlan(
        plan_id=plan_id,
        user_goal="Review PR #42",
        goal_analysis=GoalAnalysis(
            task_type=TaskType.CODE_REVIEW,
            required_capabilities=["fetch_pr_data", "synthesize_findings"],
            risk_level=RiskLevel.LOW,
            confidence=0.9,
            reasoning="Code review.",
            constraints=["read_only"],
            requires_hitl=False,
        ),
        selected_pattern="parallel_specialist",
        selected_agents=["pr_data_agent", "synthesis_agent"],
        selected_tools=["github_get_pr"],
        guardrails=[GuardrailConfig(rule_type="content_filter", config={}, reason="safety")],
        hitl_required=False,
        warnings=[],
        explanation="Parallel PR review workflow.",
        estimated_complexity="low",
        estimated_duration_seconds=35,
    )


def _make_validation(is_valid: bool = True) -> ValidationResult:
    return ValidationResult(
        is_valid=is_valid,
        errors=[] if is_valid else [ValidationError(code="MISSING_AGENT", message="Agent missing.")],
        warnings=[ValidationWarning(code="LOW_CONFIDENCE", message="Confidence below threshold.")],
    )


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


class TestPlanRepositoryCreate:
    def test_create_returns_row(self, session, repo):
        plan = _make_plan()
        validation = _make_validation()
        row = repo.create(session, plan, validation)
        session.commit()
        assert row.plan_id == "plan-001"
        assert row.goal == "Review PR #42"
        assert row.status == "pending_review"

    def test_create_default_status_is_pending_review(self, session, repo):
        row = repo.create(session, _make_plan(), _make_validation())
        session.commit()
        assert row.status == "pending_review"

    def test_create_custom_status(self, session, repo):
        row = repo.create(session, _make_plan(), _make_validation(), status="executed")
        session.commit()
        assert row.status == "executed"

    def test_create_stores_plan_json(self, session, repo):
        row = repo.create(session, _make_plan(), _make_validation())
        session.commit()
        assert "parallel_specialist" in row.plan_json

    def test_create_stores_validation_json(self, session, repo):
        validation = _make_validation(is_valid=False)
        row = repo.create(session, _make_plan(), validation)
        session.commit()
        assert "MISSING_AGENT" in row.validation_json

    def test_create_execution_run_id_is_null(self, session, repo):
        row = repo.create(session, _make_plan(), _make_validation())
        session.commit()
        assert row.execution_run_id is None

    def test_create_timestamps_are_set(self, session, repo):
        row = repo.create(session, _make_plan(), _make_validation())
        session.commit()
        assert row.created_at is not None
        assert row.updated_at is not None


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


class TestPlanRepositoryGet:
    def test_get_returns_created_row(self, session, repo):
        repo.create(session, _make_plan("plan-get-1"), _make_validation())
        session.commit()
        row = repo.get(session, "plan-get-1")
        assert row is not None
        assert row.plan_id == "plan-get-1"

    def test_get_returns_none_for_missing(self, session, repo):
        row = repo.get(session, "non-existent")
        assert row is None

    def test_get_returns_correct_goal(self, session, repo):
        repo.create(session, _make_plan("plan-goal"), _make_validation())
        session.commit()
        row = repo.get(session, "plan-goal")
        assert row.goal == "Review PR #42"


# ---------------------------------------------------------------------------
# list_all
# ---------------------------------------------------------------------------


class TestPlanRepositoryListAll:
    def test_list_all_empty_initially(self, session, repo):
        assert repo.list_all(session) == []

    def test_list_all_returns_all_plans(self, session, repo):
        repo.create(session, _make_plan("p1"), _make_validation())
        repo.create(session, _make_plan("p2"), _make_validation())
        session.commit()
        rows = repo.list_all(session)
        assert len(rows) == 2

    def test_list_all_plan_ids(self, session, repo):
        repo.create(session, _make_plan("p-a"), _make_validation())
        repo.create(session, _make_plan("p-b"), _make_validation())
        session.commit()
        ids = {row.plan_id for row in repo.list_all(session)}
        assert ids == {"p-a", "p-b"}


# ---------------------------------------------------------------------------
# update_status
# ---------------------------------------------------------------------------


class TestPlanRepositoryUpdateStatus:
    def test_update_status_changes_status(self, session, repo):
        repo.create(session, _make_plan("plan-upd"), _make_validation())
        session.commit()
        repo.update_status(session, "plan-upd", "executed")
        session.commit()
        row = repo.get(session, "plan-upd")
        assert row.status == "executed"

    def test_update_status_with_run_id(self, session, repo):
        repo.create(session, _make_plan("plan-run"), _make_validation())
        session.commit()
        repo.update_status(session, "plan-run", "executed", execution_run_id="run-xyz")
        session.commit()
        row = repo.get(session, "plan-run")
        assert row.execution_run_id == "run-xyz"

    def test_update_status_updates_updated_at(self, session, repo):
        repo.create(session, _make_plan("plan-ts"), _make_validation())
        session.commit()
        original = repo.get(session, "plan-ts").updated_at
        import time; time.sleep(0.01)
        repo.update_status(session, "plan-ts", "rejected")
        session.commit()
        updated = repo.get(session, "plan-ts").updated_at
        assert updated >= original

    def test_update_status_noop_for_missing_plan(self, session, repo):
        repo.update_status(session, "does-not-exist", "executed")
        session.commit()  # should not raise

    def test_update_status_to_rejected(self, session, repo):
        repo.create(session, _make_plan("plan-rej"), _make_validation())
        session.commit()
        repo.update_status(session, "plan-rej", "rejected")
        session.commit()
        assert repo.get(session, "plan-rej").status == "rejected"

    def test_update_status_without_run_id_keeps_existing(self, session, repo):
        repo.create(session, _make_plan("plan-nid"), _make_validation())
        session.commit()
        repo.update_status(session, "plan-nid", "executed", execution_run_id="run-1")
        session.commit()
        repo.update_status(session, "plan-nid", "failed")
        session.commit()
        row = repo.get(session, "plan-nid")
        assert row.execution_run_id == "run-1"

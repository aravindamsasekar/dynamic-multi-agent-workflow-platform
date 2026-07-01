"""SQLAlchemy ORM models for workflow execution persistence."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from platform.persistence.database import Base


class WorkflowRunRow(Base):
    __tablename__ = "workflow_runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    workflow_id: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    input: Mapped[str] = mapped_column(Text, nullable=False)
    output: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class AgentResultRow(Base):
    __tablename__ = "agent_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("workflow_runs.run_id"), nullable=False)
    agent_id: Mapped[str] = mapped_column(String, nullable=False)
    output: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ToolCallRow(Base):
    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("workflow_runs.run_id"), nullable=False)
    tool_name: Mapped[str] = mapped_column(String, nullable=False)
    input: Mapped[str | None] = mapped_column(Text, nullable=True)
    output: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_error: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class EventRow(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("workflow_runs.run_id"), nullable=False)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    payload: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class GeneratedPlanRow(Base):
    __tablename__ = "generated_plans"

    plan_id: Mapped[str] = mapped_column(String, primary_key=True)
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)   # pending_review|executed|rejected|failed
    plan_json: Mapped[str] = mapped_column(Text, nullable=False)
    validation_json: Mapped[str] = mapped_column(Text, nullable=False)
    execution_run_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class KnowledgeChunkRow(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    collection: Mapped[str] = mapped_column(String, nullable=False, index=True)
    source_file: Mapped[str] = mapped_column(String, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_hash: Mapped[str] = mapped_column(String, nullable=False)


class InstalledPackageRow(Base):
    __tablename__ = "installed_packages"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # package_id / extension_id
    version: Mapped[str] = mapped_column(String, nullable=False)
    installed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    auto_installed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    permissions_granted: Mapped[str] = mapped_column(Text, nullable=False)  # JSON array
    status: Mapped[str] = mapped_column(String, nullable=False, default="active")


class InstallHistoryRow(Base):
    __tablename__ = "install_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    package_id: Mapped[str] = mapped_column(String, nullable=False)
    action: Mapped[str] = mapped_column(String, nullable=False)   # "install"
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    permissions: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

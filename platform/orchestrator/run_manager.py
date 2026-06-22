"""RunManager — tracks WorkflowRun lifecycle."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime

from platform.core.exceptions import RunNotFound
from platform.core.models.workflow import RunStatus, WorkflowRun


class RunManager:
    """In-memory store for WorkflowRun state and lifecycle transitions.

    Valid status transitions:
        PENDING → RUNNING → COMPLETED
                           → FAILED
                           → WAITING_APPROVAL → RUNNING  (on approve)
                                              → FAILED    (on reject)
    """

    def __init__(self) -> None:
        self._runs: dict[str, WorkflowRun] = {}
        self._events: dict[str, asyncio.Event] = {}

    def create_run(self, workflow_id: str, input: str) -> WorkflowRun:
        """Create a new WorkflowRun with PENDING status and return it."""
        run = WorkflowRun(
            run_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            input=input,
        )
        self._runs[run.run_id] = run
        return run

    def get_run(self, run_id: str) -> WorkflowRun:
        """Return the WorkflowRun for the given run_id. Raises RunNotFound if missing."""
        if run_id not in self._runs:
            raise RunNotFound(f"Run '{run_id}' not found")
        return self._runs[run_id]

    def update_status(self, run_id: str, status: RunStatus) -> None:
        """Transition run to the given status and update updated_at."""
        run = self.get_run(run_id)
        run.status = status
        run.updated_at = datetime.utcnow()

    async def pause(self, run_id: str) -> None:
        """Set status to WAITING_APPROVAL and block the caller until resumed."""
        event = asyncio.Event()
        self._events[run_id] = event
        self.update_status(run_id, RunStatus.WAITING_APPROVAL)
        await event.wait()

    def resume(self, run_id: str) -> None:
        """Signal the paused run to continue, transitioning to RUNNING."""
        self.update_status(run_id, RunStatus.RUNNING)
        event = self._events.pop(run_id, None)
        if event:
            event.set()

    def complete(self, run_id: str, output: str) -> None:
        """Mark run COMPLETED and store the final output."""
        self.get_run(run_id).output = output
        self.update_status(run_id, RunStatus.COMPLETED)

    def fail(self, run_id: str, error: str) -> None:
        """Mark run FAILED and store the error message."""
        self.get_run(run_id).error = error
        self.update_status(run_id, RunStatus.FAILED)
        event = self._events.pop(run_id, None)
        if event:
            event.set()

    def list_runs(self) -> list[WorkflowRun]:
        """Return all tracked WorkflowRuns."""
        return list(self._runs.values())

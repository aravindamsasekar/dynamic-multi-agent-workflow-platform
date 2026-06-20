"""RunManager — tracks WorkflowRun lifecycle."""

from __future__ import annotations

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
        # TODO: asyncio.Event per run_id for HITL pause/resume signalling

    def create_run(self, workflow_id: str, input: str) -> WorkflowRun:
        """Create a new WorkflowRun with PENDING status and return it."""
        # TODO: implement
        raise NotImplementedError

    def get_run(self, run_id: str) -> WorkflowRun:
        """Return the WorkflowRun for the given run_id. Raises WorkflowNotFound if missing."""
        # TODO: implement
        raise NotImplementedError

    def update_status(self, run_id: str, status: RunStatus) -> None:
        """Transition run to the given status and update updated_at."""
        # TODO: implement
        raise NotImplementedError

    async def pause(self, run_id: str) -> None:
        """Set status to WAITING_APPROVAL and block the caller until resumed."""
        # TODO: implement
        raise NotImplementedError

    def resume(self, run_id: str) -> None:
        """Signal the paused run to continue, transitioning to RUNNING."""
        # TODO: implement
        raise NotImplementedError

    def complete(self, run_id: str, output: str) -> None:
        """Mark run COMPLETED and store the final output."""
        # TODO: implement
        raise NotImplementedError

    def fail(self, run_id: str, error: str) -> None:
        """Mark run FAILED and store the error message."""
        # TODO: implement
        raise NotImplementedError

    def list_runs(self) -> list[WorkflowRun]:
        """Return all tracked WorkflowRuns."""
        # TODO: implement
        raise NotImplementedError

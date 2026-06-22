"""ApprovalManager — pauses runs pending human approval."""

from __future__ import annotations

from typing import Any

from platform.core.exceptions import HITLRejected
from platform.core.interfaces.hitl import IHumanApproval


class ApprovalManager(IHumanApproval):
    """Manages HITL approval requests for paused workflow runs.

    When a pattern executor reaches a HITL checkpoint:
    1. request_approval() stores context and calls run_manager.pause(run_id)
    2. The run suspends until the API receives an approve or reject call
    3. approve() calls run_manager.resume(run_id) to continue execution
    4. reject() calls run_manager.fail(run_id) and raises HITLRejected
    """

    def __init__(self, run_manager: object) -> None:
        self._run_manager = run_manager
        self._pending: dict[str, dict[str, Any]] = {}

    async def request_approval(self, run_id: str, context: dict[str, Any]) -> None:
        """Store approval context and pause execution until a decision is made."""
        self._pending[run_id] = context
        await self._run_manager.pause(run_id)  # type: ignore[attr-defined]

    def approve(self, run_id: str, comment: str = "") -> None:
        """Resume a paused run, allowing execution to continue."""
        self._pending.pop(run_id, None)
        self._run_manager.resume(run_id)  # type: ignore[attr-defined]

    def reject(self, run_id: str, reason: str = "") -> None:
        """Fail a paused run and raise HITLRejected to abort execution."""
        self._pending.pop(run_id, None)
        self._run_manager.fail(run_id, reason or "Rejected by human reviewer")  # type: ignore[attr-defined]
        raise HITLRejected(reason)

    def get_pending(self, run_id: str) -> dict[str, Any] | None:
        """Return the pending approval context for run_id, or None if not pending."""
        return self._pending.get(run_id)

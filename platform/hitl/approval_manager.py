"""ApprovalManager — pauses runs pending human approval."""

from __future__ import annotations

from typing import Any

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
        self._pending: dict[str, dict[str, Any]] = {}  # run_id → approval context

    async def request_approval(self, run_id: str, context: dict[str, Any]) -> None:
        # TODO: implement
        raise NotImplementedError

    def approve(self, run_id: str, comment: str = "") -> None:
        # TODO: implement
        raise NotImplementedError

    def reject(self, run_id: str, reason: str = "") -> None:
        # TODO: implement
        raise NotImplementedError

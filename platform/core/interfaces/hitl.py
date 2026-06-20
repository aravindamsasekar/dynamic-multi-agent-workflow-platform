"""IHumanApproval interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class IHumanApproval(ABC):
    """Abstract interface for human-in-the-loop approval."""

    @abstractmethod
    async def request_approval(self, run_id: str, context: dict[str, Any]) -> None:
        """Pause execution and request human approval."""
        ...

    @abstractmethod
    def approve(self, run_id: str, comment: str = "") -> None:
        """Approve a paused run, allowing it to continue."""
        ...

    @abstractmethod
    def reject(self, run_id: str, reason: str = "") -> None:
        """Reject a paused run, marking it as failed."""
        ...

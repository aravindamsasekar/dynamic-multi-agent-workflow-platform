"""IObserver interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from platform.core.models.events import WorkflowEvent


class IObserver(ABC):
    """Abstract interface for workflow event observation."""

    @abstractmethod
    def on_event(self, event: WorkflowEvent) -> None:
        """Handle a workflow event."""
        ...

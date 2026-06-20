"""ConsoleObserver — emits structured workflow events to stdout."""

from __future__ import annotations

from platform.core.interfaces.observer import IObserver
from platform.core.models.events import WorkflowEvent


class ConsoleObserver(IObserver):
    """Writes workflow events as JSON lines to stdout.

    Default IObserver implementation for V1. Suitable for local development
    and demo runs. Replace with a file or external sink observer for production.
    """

    def on_event(self, event: WorkflowEvent) -> None:
        # TODO: serialize event to JSON and print to stdout
        raise NotImplementedError

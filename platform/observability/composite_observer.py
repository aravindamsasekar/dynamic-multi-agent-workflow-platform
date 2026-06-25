"""CompositeObserver — fans out events to multiple IObserver instances."""

from __future__ import annotations

from platform.core.interfaces.observer import IObserver
from platform.core.models.events import WorkflowEvent


class CompositeObserver(IObserver):
    def __init__(self, observers: list[IObserver]) -> None:
        self._observers = observers

    def on_event(self, event: WorkflowEvent) -> None:
        for observer in self._observers:
            observer.on_event(event)

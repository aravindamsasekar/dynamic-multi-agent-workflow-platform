"""IMemoryStore interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from platform.core.models.message import Message


class IMemoryStore(ABC):
    """Abstract interface for agent conversation history storage."""

    @abstractmethod
    def get_history(self, run_id: str, agent_id: str) -> list[Message]:
        """Return conversation history for a given run and agent."""
        ...

    @abstractmethod
    def append(self, run_id: str, agent_id: str, message: Message) -> None:
        """Append a message to the conversation history."""
        ...

    @abstractmethod
    def clear(self, run_id: str, agent_id: str) -> None:
        """Clear conversation history for a given run and agent."""
        ...

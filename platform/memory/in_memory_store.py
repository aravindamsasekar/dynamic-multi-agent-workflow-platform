"""InMemoryStore — conversation history storage for agent turns."""

from __future__ import annotations

from platform.core.interfaces.memory import IMemoryStore
from platform.core.models.message import Message


class InMemoryStore(IMemoryStore):
    """Stores conversation history in a dict keyed by (run_id, agent_id).

    Used by AgentRuntime to accumulate messages across the LLM → tool → LLM loop
    within a single agent execution. Scoped per run so concurrent runs are isolated.
    """

    def __init__(self) -> None:
        self._histories: dict[tuple[str, str], list[Message]] = {}

    def get_history(self, run_id: str, agent_id: str) -> list[Message]:
        """Return a copy of conversation history for the given run and agent."""
        return list(self._histories.get((run_id, agent_id), []))

    def append(self, run_id: str, agent_id: str, message: Message) -> None:
        """Append a message to the conversation history for the given run and agent."""
        key = (run_id, agent_id)
        if key not in self._histories:
            self._histories[key] = []
        self._histories[key].append(message)

    def clear(self, run_id: str, agent_id: str) -> None:
        """Clear conversation history for the given run and agent."""
        self._histories.pop((run_id, agent_id), None)

    def clear_run(self, run_id: str) -> None:
        """Clear conversation history for all agents in the given run."""
        keys_to_remove = [k for k in self._histories if k[0] == run_id]
        for key in keys_to_remove:
            del self._histories[key]

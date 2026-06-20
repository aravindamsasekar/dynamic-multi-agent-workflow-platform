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
        # TODO: implement
        raise NotImplementedError

    def append(self, run_id: str, agent_id: str, message: Message) -> None:
        # TODO: implement
        raise NotImplementedError

    def clear(self, run_id: str, agent_id: str) -> None:
        # TODO: implement
        raise NotImplementedError

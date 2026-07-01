"""AgentRegistry — indexes AgentDefinitions by agent_id."""

from __future__ import annotations

from platform.core.exceptions import AgentNotFound
from platform.core.models.agent import AgentDefinition


class AgentRegistry:
    """In-memory registry for AgentDefinitions.

    Populated at startup by ConfigLoader. Queried by pattern executors.
    """

    def __init__(self) -> None:
        self._store: dict[str, AgentDefinition] = {}

    def register(self, definition: AgentDefinition) -> None:
        """Add an AgentDefinition to the registry."""
        self._store[definition.agent_id] = definition

    def get(self, agent_id: str) -> AgentDefinition:
        """Return AgentDefinition by id. Raises AgentNotFound if missing."""
        if agent_id not in self._store:
            raise AgentNotFound(f"Agent '{agent_id}' not found")
        return self._store[agent_id]

    def list_all(self) -> list[AgentDefinition]:
        """Return all registered AgentDefinitions."""
        return list(self._store.values())

    def exists(self, agent_id: str) -> bool:
        """Return True if an agent with the given id is registered."""
        return agent_id in self._store

    def unregister(self, agent_id: str) -> None:
        """Remove an AgentDefinition by id. No-op if not present."""
        self._store.pop(agent_id, None)

    def clear(self) -> None:
        """Remove all registered AgentDefinitions."""
        self._store.clear()

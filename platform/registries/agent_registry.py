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
        # TODO: implement
        raise NotImplementedError

    def get(self, agent_id: str) -> AgentDefinition:
        """Return AgentDefinition by id. Raises AgentNotFound if missing."""
        # TODO: implement
        raise NotImplementedError

    def list_all(self) -> list[AgentDefinition]:
        """Return all registered AgentDefinitions."""
        # TODO: implement
        raise NotImplementedError

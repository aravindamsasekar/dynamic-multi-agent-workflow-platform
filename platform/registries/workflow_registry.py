"""WorkflowRegistry — indexes WorkflowDefinitions by workflow_id."""

from __future__ import annotations

from platform.core.exceptions import WorkflowNotFound
from platform.core.models.workflow import WorkflowDefinition


class WorkflowRegistry:
    """In-memory registry for WorkflowDefinitions.

    Populated at startup by ConfigLoader. Queried by Orchestrator.
    """

    def __init__(self) -> None:
        self._store: dict[str, WorkflowDefinition] = {}

    def register(self, definition: WorkflowDefinition) -> None:
        """Add a WorkflowDefinition to the registry."""
        # TODO: implement
        raise NotImplementedError

    def get(self, workflow_id: str) -> WorkflowDefinition:
        """Return WorkflowDefinition by id. Raises WorkflowNotFound if missing."""
        # TODO: implement
        raise NotImplementedError

    def list_all(self) -> list[WorkflowDefinition]:
        """Return all registered WorkflowDefinitions."""
        # TODO: implement
        raise NotImplementedError

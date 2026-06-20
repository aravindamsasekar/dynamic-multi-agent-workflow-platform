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
        self._store[definition.workflow_id] = definition

    def get(self, workflow_id: str) -> WorkflowDefinition:
        """Return WorkflowDefinition by id. Raises WorkflowNotFound if missing."""
        if workflow_id not in self._store:
            raise WorkflowNotFound(f"Workflow '{workflow_id}' not found")
        return self._store[workflow_id]

    def list_all(self) -> list[WorkflowDefinition]:
        """Return all registered WorkflowDefinitions."""
        return list(self._store.values())

    def exists(self, workflow_id: str) -> bool:
        """Return True if a workflow with the given id is registered."""
        return workflow_id in self._store

    def clear(self) -> None:
        """Remove all registered WorkflowDefinitions."""
        self._store.clear()

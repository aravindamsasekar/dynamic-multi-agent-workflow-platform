"""IPatternExecutor — the single contract for all pattern executors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from platform.core.models.context import ExecutionContext
from platform.core.models.workflow import WorkflowResult


class IPatternExecutor(ABC):
    """Abstract base class for all workflow pattern executors.

    Each concrete executor implements one supported workflow pattern.
    The Orchestrator selects the correct executor based on
    WorkflowDefinition.pattern and calls execute(context).
    """

    @abstractmethod
    async def execute(self, context: ExecutionContext, workflow_input: str) -> WorkflowResult:
        """Execute the pattern and return the final workflow result."""
        ...

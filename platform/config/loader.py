"""ConfigLoader — scans the workflows directory and populates registries at startup."""

from __future__ import annotations

from pathlib import Path


class ConfigLoader:
    """Reads workflow YAML files and populates the three registries.

    Called once during application startup in api/main.py.
    For each subdirectory under workflows_dir, loads:
        workflow.yaml  → WorkflowRegistry
        agents.yaml    → AgentRegistry
        tools.yaml     → ToolRegistry (instantiates the correct IToolAdapter per entry)
    """

    def __init__(
        self,
        workflow_registry: object,
        agent_registry: object,
        tool_registry: object,
    ) -> None:
        self._workflow_registry = workflow_registry
        self._agent_registry = agent_registry
        self._tool_registry = tool_registry

    def load_all(self, workflows_dir: Path) -> None:
        """Scan workflows_dir and load all workflow definitions into registries."""
        # TODO: implement
        raise NotImplementedError

    def _load_workflow(self, workflow_dir: Path) -> None:
        """Load a single workflow directory (workflow.yaml + agents.yaml + tools.yaml)."""
        # TODO: implement
        raise NotImplementedError

    def _load_agents(self, agents_path: Path) -> None:
        """Parse agents.yaml and register AgentDefinitions in AgentRegistry."""
        # TODO: implement
        raise NotImplementedError

    def _load_tools(self, tools_path: Path) -> None:
        """Parse tools.yaml, instantiate the correct adapter, register in ToolRegistry."""
        # TODO: implement
        raise NotImplementedError

"""ConfigLoader — scans the workflows directory and populates registries at startup."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import sys

import yaml

from platform.config.validator import ConfigValidator
from platform.core.interfaces.tool import IToolAdapter
from platform.core.models.agent import AgentDefinition
from platform.core.models.tool import AdapterType, ToolDefinition
from platform.core.models.workflow import WorkflowDefinition
from platform.tools.github_adapter import GitHubAdapter
from platform.tools.http_adapter import HTTPAdapter
from platform.tools.mcp_adapter import MCPAdapter
from platform.tools.mock_adapter import MockAdapter

_AdapterBuilder = Callable[[dict[str, Any]], IToolAdapter]


def _build_mock_adapter(cfg: dict[str, Any]) -> IToolAdapter:
    return MockAdapter(
        response=cfg.get("response", ""),
        is_error=bool(cfg.get("is_error", False)),
    )


def _build_http_adapter(cfg: dict[str, Any]) -> IToolAdapter:
    return HTTPAdapter(
        url=cfg["url"],
        method=cfg.get("method", "POST"),
        headers=cfg.get("headers"),
    )


def _build_mcp_adapter(cfg: dict[str, Any]) -> IToolAdapter:
    return MCPAdapter(server_url=cfg["server_url"])


def _build_github_adapter(cfg: dict[str, Any]) -> IToolAdapter:
    return GitHubAdapter(operation=cfg["operation"])


_ADAPTER_BUILDERS: dict[AdapterType, _AdapterBuilder] = {
    AdapterType.MOCK: _build_mock_adapter,
    AdapterType.HTTP: _build_http_adapter,
    AdapterType.MCP: _build_mcp_adapter,
    AdapterType.GITHUB: _build_github_adapter,
}


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
        self._validator = ConfigValidator()

    def load_all(self, workflows_dir: Path) -> None:
        """Scan workflows_dir and load all workflow definitions into registries."""
        for workflow_dir in sorted(workflows_dir.iterdir()):
            if workflow_dir.is_dir():
                try:
                    self._load_workflow(workflow_dir)
                except Exception as exc:
                    print(
                        f"[WARNING] Skipping '{workflow_dir.name}': {exc}",
                        file=sys.stderr,
                    )

    def load_one(self, workflow_dir: Path) -> None:
        """Load a single workflow directory into registries."""
        self._load_workflow(workflow_dir)

    def _load_workflow(self, workflow_dir: Path) -> None:
        """Load a single workflow directory (workflow.yaml + agents.yaml + tools.yaml)."""
        workflow_path = workflow_dir / "workflow.yaml"
        raw: dict[str, Any] = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        self._validator.validate_workflow(raw, source=str(workflow_path))
        self._workflow_registry.register(WorkflowDefinition(**raw))  # type: ignore[attr-defined]

        self._load_agents(workflow_dir / "agents.yaml")
        self._load_tools(workflow_dir / "tools.yaml")

    def _load_agents(self, agents_path: Path) -> None:
        """Parse agents.yaml and register AgentDefinitions in AgentRegistry."""
        raw: dict[str, Any] = yaml.safe_load(agents_path.read_text(encoding="utf-8"))
        self._validator.validate_agents(raw, source=str(agents_path))
        for agent_data in (raw["agents"] or []):
            self._agent_registry.register(AgentDefinition(**agent_data))  # type: ignore[attr-defined]

    def _load_tools(self, tools_path: Path) -> None:
        """Parse tools.yaml, instantiate the correct adapter, register in ToolRegistry."""
        raw: dict[str, Any] = yaml.safe_load(tools_path.read_text(encoding="utf-8"))
        self._validator.validate_tools(raw, source=str(tools_path))
        for tool_data in (raw["tools"] or []):
            tool_def = ToolDefinition(**tool_data)
            adapter = _ADAPTER_BUILDERS[tool_def.adapter_type](tool_def.adapter_config)
            self._tool_registry.register(tool_def.name, adapter, tool_def)  # type: ignore[attr-defined]

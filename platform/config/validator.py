"""ConfigValidator — validates workflow YAML files against expected schemas."""

from __future__ import annotations

from typing import Any

from platform.core.exceptions import ConfigValidationError
from platform.core.models.tool import AdapterType
from platform.core.models.workflow import PatternType


class ConfigValidator:
    """Validates parsed YAML dicts against expected schemas before loading into registries.

    Provides readable error messages on schema violations so misconfigured
    workflow files fail fast with actionable output at startup.
    """

    def validate_workflow(self, data: dict[str, Any], source: str = "") -> None:
        """Validate a parsed workflow.yaml dict. Raises ConfigValidationError on failure."""
        prefix = f"[{source}] " if source else ""
        if not isinstance(data, dict):
            raise ConfigValidationError(
                f"{prefix}workflow.yaml must be a YAML mapping, got {type(data).__name__}"
            )
        for field in ("workflow_id", "name", "pattern"):
            if field not in data:
                raise ConfigValidationError(f"{prefix}Missing required field: '{field}'")
            if not isinstance(data[field], str) or not data[field].strip():
                raise ConfigValidationError(
                    f"{prefix}Field '{field}' must be a non-empty string"
                )
        valid_patterns = {p.value for p in PatternType}
        if data["pattern"] not in valid_patterns:
            raise ConfigValidationError(
                f"{prefix}Invalid pattern '{data['pattern']}'. "
                f"Must be one of: {sorted(valid_patterns)}"
            )

    def validate_agents(self, data: dict[str, Any], source: str = "") -> None:
        """Validate a parsed agents.yaml dict. Raises ConfigValidationError on failure."""
        prefix = f"[{source}] " if source else ""
        if not isinstance(data, dict) or "agents" not in data:
            raise ConfigValidationError(
                f"{prefix}agents.yaml must have a top-level 'agents' key"
            )
        agents = data["agents"] or []
        if not isinstance(agents, list):
            raise ConfigValidationError(f"{prefix}'agents' must be a list")
        for i, agent in enumerate(agents):
            if not isinstance(agent, dict):
                raise ConfigValidationError(f"{prefix}Agent at index {i} must be a mapping")
            for field in ("agent_id", "name", "system_prompt"):
                if field not in agent:
                    raise ConfigValidationError(
                        f"{prefix}Agent at index {i} missing required field: '{field}'"
                    )
                if not isinstance(agent[field], str) or not agent[field].strip():
                    raise ConfigValidationError(
                        f"{prefix}Agent at index {i} field '{field}' must be a non-empty string"
                    )

    def validate_tools(self, data: dict[str, Any], source: str = "") -> None:
        """Validate a parsed tools.yaml dict. Raises ConfigValidationError on failure."""
        prefix = f"[{source}] " if source else ""
        if not isinstance(data, dict) or "tools" not in data:
            raise ConfigValidationError(
                f"{prefix}tools.yaml must have a top-level 'tools' key"
            )
        tools = data["tools"] or []
        if not isinstance(tools, list):
            raise ConfigValidationError(f"{prefix}'tools' must be a list")
        valid_types = {t.value for t in AdapterType}
        for i, tool in enumerate(tools):
            if not isinstance(tool, dict):
                raise ConfigValidationError(f"{prefix}Tool at index {i} must be a mapping")
            for field in ("name", "description", "input_schema", "adapter_type"):
                if field not in tool:
                    raise ConfigValidationError(
                        f"{prefix}Tool at index {i} missing required field: '{field}'"
                    )
            name = tool.get("name", f"index {i}")
            if tool["adapter_type"] not in valid_types:
                raise ConfigValidationError(
                    f"{prefix}Tool '{name}' has invalid adapter_type '{tool['adapter_type']}'. "
                    f"Must be one of: {sorted(valid_types)}"
                )
            cfg: dict[str, Any] = tool.get("adapter_config") or {}
            if tool["adapter_type"] == AdapterType.HTTP.value and "url" not in cfg:
                raise ConfigValidationError(
                    f"{prefix}Tool '{name}' (http) missing adapter_config.url"
                )
            if tool["adapter_type"] == AdapterType.MCP.value and "server_url" not in cfg:
                raise ConfigValidationError(
                    f"{prefix}Tool '{name}' (mcp) missing adapter_config.server_url"
                )
            if tool["adapter_type"] == AdapterType.GITHUB.value and "operation" not in cfg:
                raise ConfigValidationError(
                    f"{prefix}Tool '{name}' (github) missing adapter_config.operation"
                )

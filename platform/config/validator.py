"""ConfigValidator — validates workflow YAML files against expected schemas."""

from __future__ import annotations

from typing import Any


class ConfigValidator:
    """Validates parsed YAML dicts against expected schemas before loading into registries.

    Provides readable error messages on schema violations so misconfigured
    workflow files fail fast with actionable output at startup.
    """

    def validate_workflow(self, data: dict[str, Any], source: str = "") -> None:
        """Validate a parsed workflow.yaml dict. Raises ConfigValidationError on failure."""
        # TODO: implement
        raise NotImplementedError

    def validate_agents(self, data: dict[str, Any], source: str = "") -> None:
        """Validate a parsed agents.yaml dict. Raises ConfigValidationError on failure."""
        # TODO: implement
        raise NotImplementedError

    def validate_tools(self, data: dict[str, Any], source: str = "") -> None:
        """Validate a parsed tools.yaml dict. Raises ConfigValidationError on failure."""
        # TODO: implement
        raise NotImplementedError

"""Unit tests for ConfigLoader and ConfigValidator."""

import pytest


class TestConfigLoader:
    # TODO: test load_all() populates WorkflowRegistry, AgentRegistry, ToolRegistry
    # TODO: test load_all() skips non-directory entries in workflows_dir
    # TODO: test _load_tools() instantiates MockAdapter for adapter_type: mock
    # TODO: test _load_tools() instantiates HTTPAdapter for adapter_type: http
    # TODO: test _load_tools() instantiates MCPAdapter for adapter_type: mcp
    pass


class TestConfigValidator:
    # TODO: test validate_workflow() passes on valid workflow dict
    # TODO: test validate_workflow() raises ConfigValidationError on missing required fields
    # TODO: test validate_agents() passes on valid agents dict
    # TODO: test validate_tools() raises ConfigValidationError on unknown adapter_type
    pass

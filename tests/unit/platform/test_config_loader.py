"""Unit tests for ConfigLoader and ConfigValidator."""

from __future__ import annotations

from pathlib import Path

import pytest

from unittest.mock import MagicMock

from platform.config.loader import ConfigLoader
from platform.config.validator import ConfigValidator
from platform.core.exceptions import ConfigValidationError
from platform.knowledge.service import KnowledgeService
from platform.registries.agent_registry import AgentRegistry
from platform.registries.tool_registry import ToolRegistry
from platform.registries.workflow_registry import WorkflowRegistry
from platform.tools.http_adapter import HTTPAdapter
from platform.tools.knowledge_adapter import KnowledgeAdapter
from platform.tools.mcp_adapter import MCPAdapter
from platform.tools.mock_adapter import MockAdapter


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_MINIMAL_WORKFLOW = """\
workflow_id: test-wf
name: Test Workflow
pattern: router
"""

_MINIMAL_AGENTS = """\
agents:
  - agent_id: test-agent
    name: Test Agent
    system_prompt: You are a test agent.
"""

_EMPTY_TOOLS = "tools: []\n"

_MOCK_TOOL = """\
tools:
  - name: test_tool
    description: A test tool
    input_schema:
      type: object
      properties: {}
    adapter_type: mock
    adapter_config:
      response: mock result
      is_error: false
"""

_HTTP_TOOL = """\
tools:
  - name: http_tool
    description: An HTTP tool
    input_schema:
      type: object
      properties: {}
    adapter_type: http
    adapter_config:
      url: https://example.com/api
      method: POST
"""

_MCP_TOOL = """\
tools:
  - name: mcp_tool
    description: An MCP tool
    input_schema:
      type: object
      properties: {}
    adapter_type: mcp
    adapter_config:
      server_command: npx
      server_args:
        - -y
        - "@modelcontextprotocol/server-filesystem"
        - "."
      tool_name: read_file
"""


def _make_workflow_dir(
    tmp_path: Path,
    wf_id: str,
    *,
    workflow_yaml: str = _MINIMAL_WORKFLOW,
    agents_yaml: str = _MINIMAL_AGENTS,
    tools_yaml: str = _EMPTY_TOOLS,
) -> Path:
    d = tmp_path / wf_id
    d.mkdir()
    (d / "workflow.yaml").write_text(workflow_yaml, encoding="utf-8")
    (d / "agents.yaml").write_text(agents_yaml, encoding="utf-8")
    (d / "tools.yaml").write_text(tools_yaml, encoding="utf-8")
    return d


def _make_loader(
    tmp_path: Path,
    knowledge_service: object | None = None,
) -> tuple[ConfigLoader, WorkflowRegistry, AgentRegistry, ToolRegistry]:
    wf_reg = WorkflowRegistry()
    ag_reg = AgentRegistry()
    tl_reg = ToolRegistry()
    return (
        ConfigLoader(wf_reg, ag_reg, tl_reg, knowledge_service=knowledge_service),
        wf_reg,
        ag_reg,
        tl_reg,
    )


# ---------------------------------------------------------------------------
# TestConfigLoader
# ---------------------------------------------------------------------------


class TestConfigLoader:
    def test_load_all_registers_workflow(self, tmp_path: Path) -> None:
        _make_workflow_dir(tmp_path, "test-wf")
        loader, wf_reg, _, _ = _make_loader(tmp_path)
        loader.load_all(tmp_path)

        wf = wf_reg.get("test-wf")
        assert wf.workflow_id == "test-wf"
        assert wf.name == "Test Workflow"

    def test_load_all_registers_agents(self, tmp_path: Path) -> None:
        _make_workflow_dir(tmp_path, "test-wf")
        loader, _, ag_reg, _ = _make_loader(tmp_path)
        loader.load_all(tmp_path)

        agent = ag_reg.get("test-agent")
        assert agent.agent_id == "test-agent"
        assert agent.name == "Test Agent"
        assert agent.system_prompt == "You are a test agent."

    def test_load_all_registers_mock_tool(self, tmp_path: Path) -> None:
        _make_workflow_dir(tmp_path, "test-wf", tools_yaml=_MOCK_TOOL)
        loader, _, _, tl_reg = _make_loader(tmp_path)
        loader.load_all(tmp_path)

        adapter = tl_reg.get("test_tool")
        assert isinstance(adapter, MockAdapter)

    def test_load_all_registers_http_tool(self, tmp_path: Path) -> None:
        _make_workflow_dir(tmp_path, "test-wf", tools_yaml=_HTTP_TOOL)
        loader, _, _, tl_reg = _make_loader(tmp_path)
        loader.load_all(tmp_path)

        adapter = tl_reg.get("http_tool")
        assert isinstance(adapter, HTTPAdapter)

    def test_load_all_registers_mcp_tool(self, tmp_path: Path) -> None:
        _make_workflow_dir(tmp_path, "test-wf", tools_yaml=_MCP_TOOL)
        loader, _, _, tl_reg = _make_loader(tmp_path)
        loader.load_all(tmp_path)

        adapter = tl_reg.get("mcp_tool")
        assert isinstance(adapter, MCPAdapter)
        assert adapter._tool_name == "read_file"

    def test_load_all_tool_definition_also_registered(self, tmp_path: Path) -> None:
        _make_workflow_dir(tmp_path, "test-wf", tools_yaml=_MOCK_TOOL)
        loader, _, _, tl_reg = _make_loader(tmp_path)
        loader.load_all(tmp_path)

        tool_def = tl_reg.get_definition("test_tool")
        assert tool_def is not None
        assert tool_def.name == "test_tool"
        assert tool_def.description == "A test tool"

    def test_load_all_multiple_workflows(self, tmp_path: Path) -> None:
        _make_workflow_dir(
            tmp_path, "wf-a",
            workflow_yaml="workflow_id: wf-a\nname: Workflow A\npattern: router\n",
        )
        _make_workflow_dir(
            tmp_path, "wf-b",
            workflow_yaml="workflow_id: wf-b\nname: Workflow B\npattern: parallel_specialist\n",
        )
        loader, wf_reg, _, _ = _make_loader(tmp_path)
        loader.load_all(tmp_path)

        assert wf_reg.exists("wf-a")
        assert wf_reg.exists("wf-b")
        assert len(wf_reg.list_all()) == 2

    def test_load_all_empty_tools_list(self, tmp_path: Path) -> None:
        _make_workflow_dir(tmp_path, "test-wf", tools_yaml=_EMPTY_TOOLS)
        loader, _, _, tl_reg = _make_loader(tmp_path)
        loader.load_all(tmp_path)  # must not raise

        assert tl_reg.list_all() == []

    def test_load_all_empty_agents_list(self, tmp_path: Path) -> None:
        _make_workflow_dir(tmp_path, "test-wf", agents_yaml="agents: []\n")
        loader, _, ag_reg, _ = _make_loader(tmp_path)
        loader.load_all(tmp_path)  # must not raise

        assert ag_reg.list_all() == []

    def test_load_all_skips_invalid_workflow_with_warning(
        self, tmp_path: Path, capsys
    ) -> None:
        _make_workflow_dir(
            tmp_path, "bad-wf",
            workflow_yaml="name: Missing ID\npattern: router\n",  # no workflow_id
        )
        loader, wf_reg, _, _ = _make_loader(tmp_path)
        loader.load_all(tmp_path)  # must not raise
        assert wf_reg.list_all() == []  # invalid dir was skipped
        assert "bad-wf" in capsys.readouterr().err

    def test_load_all_skips_non_directory_entries(self, tmp_path: Path) -> None:
        _make_workflow_dir(tmp_path, "test-wf")
        (tmp_path / "stray_file.yaml").write_text("ignored", encoding="utf-8")
        loader, wf_reg, _, _ = _make_loader(tmp_path)
        loader.load_all(tmp_path)  # must not raise on stray file

        assert len(wf_reg.list_all()) == 1

    def test_load_all_agent_optional_fields_have_defaults(self, tmp_path: Path) -> None:
        agents_yaml = """\
agents:
  - agent_id: minimal-agent
    name: Minimal
    system_prompt: Minimal prompt.
"""
        _make_workflow_dir(tmp_path, "test-wf", agents_yaml=agents_yaml)
        loader, _, ag_reg, _ = _make_loader(tmp_path)
        loader.load_all(tmp_path)

        agent = ag_reg.get("minimal-agent")
        assert agent.tool_names == []
        assert agent.llm_config.model == "claude-sonnet-4-6"

    def test_load_all_mock_adapter_config_applied(self, tmp_path: Path) -> None:
        tools_yaml = """\
tools:
  - name: my_mock
    description: configured mock
    input_schema: {type: object, properties: {}}
    adapter_type: mock
    adapter_config:
      response: "custom response"
      is_error: false
"""
        _make_workflow_dir(tmp_path, "test-wf", tools_yaml=tools_yaml)
        loader, _, _, tl_reg = _make_loader(tmp_path)
        loader.load_all(tmp_path)

        adapter = tl_reg.get("my_mock")
        assert isinstance(adapter, MockAdapter)
        assert adapter._response == "custom response"


# ---------------------------------------------------------------------------
# TestConfigValidator
# ---------------------------------------------------------------------------


class TestConfigValidator:
    def test_validate_workflow_valid_passes(self) -> None:
        v = ConfigValidator()
        v.validate_workflow({"workflow_id": "wf-1", "name": "WF", "pattern": "router"})

    def test_validate_workflow_missing_workflow_id_raises(self) -> None:
        v = ConfigValidator()
        with pytest.raises(ConfigValidationError, match="workflow_id"):
            v.validate_workflow({"name": "WF", "pattern": "router"})

    def test_validate_workflow_missing_name_raises(self) -> None:
        v = ConfigValidator()
        with pytest.raises(ConfigValidationError, match="name"):
            v.validate_workflow({"workflow_id": "wf-1", "pattern": "router"})

    def test_validate_workflow_invalid_pattern_raises(self) -> None:
        v = ConfigValidator()
        with pytest.raises(ConfigValidationError, match="sequential"):
            v.validate_workflow({"workflow_id": "wf-1", "name": "WF", "pattern": "sequential"})

    def test_validate_workflow_source_appears_in_error(self) -> None:
        v = ConfigValidator()
        with pytest.raises(ConfigValidationError, match="/some/path/workflow.yaml"):
            v.validate_workflow(
                {"name": "WF", "pattern": "router"},
                source="/some/path/workflow.yaml",
            )

    def test_validate_agents_valid_passes(self) -> None:
        v = ConfigValidator()
        v.validate_agents({
            "agents": [
                {"agent_id": "a1", "name": "Agent 1", "system_prompt": "You are a1."}
            ]
        })

    def test_validate_agents_missing_system_prompt_raises(self) -> None:
        v = ConfigValidator()
        with pytest.raises(ConfigValidationError, match="system_prompt"):
            v.validate_agents({
                "agents": [{"agent_id": "a1", "name": "Agent 1"}]
            })

    def test_validate_agents_missing_agents_key_raises(self) -> None:
        v = ConfigValidator()
        with pytest.raises(ConfigValidationError, match="'agents'"):
            v.validate_agents({"something_else": []})

    def test_validate_agents_empty_list_passes(self) -> None:
        v = ConfigValidator()
        v.validate_agents({"agents": []})

    def test_validate_tools_valid_passes(self) -> None:
        v = ConfigValidator()
        v.validate_tools({
            "tools": [{
                "name": "t1",
                "description": "desc",
                "input_schema": {"type": "object"},
                "adapter_type": "mock",
            }]
        })

    def test_validate_tools_http_missing_url_raises(self) -> None:
        v = ConfigValidator()
        with pytest.raises(ConfigValidationError, match="url"):
            v.validate_tools({
                "tools": [{
                    "name": "t1",
                    "description": "desc",
                    "input_schema": {"type": "object"},
                    "adapter_type": "http",
                    "adapter_config": {},
                }]
            })

    def test_validate_tools_mcp_missing_server_command_raises(self) -> None:
        v = ConfigValidator()
        with pytest.raises(ConfigValidationError, match="server_command"):
            v.validate_tools({
                "tools": [{
                    "name": "t1",
                    "description": "desc",
                    "input_schema": {"type": "object"},
                    "adapter_type": "mcp",
                    "adapter_config": {"tool_name": "read_file"},
                }]
            })

    def test_validate_tools_mcp_missing_tool_name_raises(self) -> None:
        v = ConfigValidator()
        with pytest.raises(ConfigValidationError, match="tool_name"):
            v.validate_tools({
                "tools": [{
                    "name": "t1",
                    "description": "desc",
                    "input_schema": {"type": "object"},
                    "adapter_type": "mcp",
                    "adapter_config": {"server_command": "npx"},
                }]
            })

    def test_validate_tools_mcp_server_args_not_list_raises(self) -> None:
        v = ConfigValidator()
        with pytest.raises(ConfigValidationError, match="server_args"):
            v.validate_tools({
                "tools": [{
                    "name": "t1",
                    "description": "desc",
                    "input_schema": {"type": "object"},
                    "adapter_type": "mcp",
                    "adapter_config": {
                        "server_command": "npx",
                        "tool_name": "read_file",
                        "server_args": "not-a-list",
                    },
                }]
            })

    def test_validate_tools_mcp_valid_passes(self) -> None:
        v = ConfigValidator()
        v.validate_tools({
            "tools": [{
                "name": "t1",
                "description": "desc",
                "input_schema": {"type": "object"},
                "adapter_type": "mcp",
                "adapter_config": {
                    "server_command": "npx",
                    "server_args": ["-y", "server"],
                    "tool_name": "read_file",
                },
            }]
        })

    def test_validate_tools_invalid_adapter_type_raises(self) -> None:
        v = ConfigValidator()
        with pytest.raises(ConfigValidationError, match="grpc"):
            v.validate_tools({
                "tools": [{
                    "name": "t1",
                    "description": "desc",
                    "input_schema": {"type": "object"},
                    "adapter_type": "grpc",
                }]
            })

    def test_validate_tools_empty_list_passes(self) -> None:
        v = ConfigValidator()
        v.validate_tools({"tools": []})

    def test_validate_tools_null_tools_value_passes(self) -> None:
        v = ConfigValidator()
        v.validate_tools({"tools": None})


# ---------------------------------------------------------------------------
# TestConfigValidatorKnowledge
# ---------------------------------------------------------------------------

_KNOWLEDGE_TOOL_BASE = {
    "name": "ks",
    "description": "Knowledge search",
    "input_schema": {"type": "object"},
    "adapter_type": "knowledge",
}


class TestConfigValidatorKnowledge:
    def _wrap(self, adapter_config: dict) -> dict:
        return {"tools": [{**_KNOWLEDGE_TOOL_BASE, "adapter_config": adapter_config}]}

    def test_valid_knowledge_config_passes(self) -> None:
        v = ConfigValidator()
        v.validate_tools(self._wrap({"collections": ["col-a"]}))

    def test_valid_with_top_k_passes(self) -> None:
        v = ConfigValidator()
        v.validate_tools(self._wrap({"collections": ["col-a"], "top_k": 5}))

    def test_missing_collections_raises(self) -> None:
        v = ConfigValidator()
        with pytest.raises(ConfigValidationError, match="collections"):
            v.validate_tools(self._wrap({}))

    def test_empty_collections_list_raises(self) -> None:
        v = ConfigValidator()
        with pytest.raises(ConfigValidationError, match="non-empty"):
            v.validate_tools(self._wrap({"collections": []}))

    def test_collections_not_a_list_raises(self) -> None:
        v = ConfigValidator()
        with pytest.raises(ConfigValidationError, match="non-empty"):
            v.validate_tools(self._wrap({"collections": "col-a"}))

    def test_collections_with_empty_string_raises(self) -> None:
        v = ConfigValidator()
        with pytest.raises(ConfigValidationError, match="non-empty strings"):
            v.validate_tools(self._wrap({"collections": ["col-a", ""]}))

    def test_collections_with_non_string_raises(self) -> None:
        v = ConfigValidator()
        with pytest.raises(ConfigValidationError, match="non-empty strings"):
            v.validate_tools(self._wrap({"collections": [123]}))

    def test_multiple_collections_passes(self) -> None:
        v = ConfigValidator()
        v.validate_tools(self._wrap({"collections": ["col-a", "col-b", "col-c"]}))

    def test_top_k_zero_raises(self) -> None:
        v = ConfigValidator()
        with pytest.raises(ConfigValidationError, match="positive integer"):
            v.validate_tools(self._wrap({"collections": ["col-a"], "top_k": 0}))

    def test_top_k_negative_raises(self) -> None:
        v = ConfigValidator()
        with pytest.raises(ConfigValidationError, match="positive integer"):
            v.validate_tools(self._wrap({"collections": ["col-a"], "top_k": -1}))

    def test_top_k_string_raises(self) -> None:
        v = ConfigValidator()
        with pytest.raises(ConfigValidationError, match="positive integer"):
            v.validate_tools(self._wrap({"collections": ["col-a"], "top_k": "five"}))

    def test_top_k_bool_raises(self) -> None:
        v = ConfigValidator()
        with pytest.raises(ConfigValidationError, match="positive integer"):
            v.validate_tools(self._wrap({"collections": ["col-a"], "top_k": True}))

    def test_top_k_omitted_passes(self) -> None:
        v = ConfigValidator()
        v.validate_tools(self._wrap({"collections": ["col-a"]}))


# ---------------------------------------------------------------------------
# TestConfigLoaderKnowledge
# ---------------------------------------------------------------------------

_KNOWLEDGE_TOOL_YAML = """\
tools:
  - name: knowledge_search
    description: Search knowledge base
    input_schema:
      type: object
      properties:
        query:
          type: string
    adapter_type: knowledge
    adapter_config:
      collections:
        - coding-standards
        - architecture
      top_k: 3
"""

_KNOWLEDGE_TOOL_NO_TOP_K_YAML = """\
tools:
  - name: knowledge_search
    description: Search knowledge base
    input_schema:
      type: object
      properties:
        query:
          type: string
    adapter_type: knowledge
    adapter_config:
      collections:
        - docs
"""


class TestConfigLoaderKnowledge:
    def _mock_service(self) -> MagicMock:
        return MagicMock(spec=KnowledgeService)

    def test_knowledge_tool_builds_knowledge_adapter(self, tmp_path: Path) -> None:
        _make_workflow_dir(tmp_path, "wf", tools_yaml=_KNOWLEDGE_TOOL_YAML)
        service = self._mock_service()
        loader, _, _, tl_reg = _make_loader(tmp_path, knowledge_service=service)
        loader.load_all(tmp_path)

        adapter = tl_reg.get("knowledge_search")
        assert isinstance(adapter, KnowledgeAdapter)

    def test_knowledge_adapter_has_correct_collections(self, tmp_path: Path) -> None:
        _make_workflow_dir(tmp_path, "wf", tools_yaml=_KNOWLEDGE_TOOL_YAML)
        service = self._mock_service()
        loader, _, _, tl_reg = _make_loader(tmp_path, knowledge_service=service)
        loader.load_all(tmp_path)

        adapter = tl_reg.get("knowledge_search")
        assert isinstance(adapter, KnowledgeAdapter)
        assert adapter._collections == ["coding-standards", "architecture"]

    def test_knowledge_adapter_has_correct_top_k(self, tmp_path: Path) -> None:
        _make_workflow_dir(tmp_path, "wf", tools_yaml=_KNOWLEDGE_TOOL_YAML)
        service = self._mock_service()
        loader, _, _, tl_reg = _make_loader(tmp_path, knowledge_service=service)
        loader.load_all(tmp_path)

        adapter = tl_reg.get("knowledge_search")
        assert isinstance(adapter, KnowledgeAdapter)
        assert adapter._top_k == 3

    def test_knowledge_adapter_default_top_k_when_omitted(self, tmp_path: Path) -> None:
        _make_workflow_dir(tmp_path, "wf", tools_yaml=_KNOWLEDGE_TOOL_NO_TOP_K_YAML)
        service = self._mock_service()
        loader, _, _, tl_reg = _make_loader(tmp_path, knowledge_service=service)
        loader.load_all(tmp_path)

        adapter = tl_reg.get("knowledge_search")
        assert isinstance(adapter, KnowledgeAdapter)
        assert adapter._top_k == 5

    def test_knowledge_adapter_receives_service(self, tmp_path: Path) -> None:
        _make_workflow_dir(tmp_path, "wf", tools_yaml=_KNOWLEDGE_TOOL_YAML)
        service = self._mock_service()
        loader, _, _, tl_reg = _make_loader(tmp_path, knowledge_service=service)
        loader.load_all(tmp_path)

        adapter = tl_reg.get("knowledge_search")
        assert isinstance(adapter, KnowledgeAdapter)
        assert adapter._service is service

    def test_missing_service_raises_clear_error(self, tmp_path: Path) -> None:
        """load_one() propagates the ConfigValidationError directly."""
        _make_workflow_dir(tmp_path, "wf", tools_yaml=_KNOWLEDGE_TOOL_YAML)
        loader, _, _, _ = _make_loader(tmp_path, knowledge_service=None)
        wf_dir = tmp_path / "wf"
        with pytest.raises(ConfigValidationError, match="KnowledgeService"):
            loader.load_one(wf_dir)

    def test_missing_service_load_all_logs_warning(self, tmp_path: Path) -> None:
        """load_all() swallows the error as a warning (existing behaviour for all tool errors)."""
        _make_workflow_dir(tmp_path, "wf", tools_yaml=_KNOWLEDGE_TOOL_YAML)
        loader, _, _, _ = _make_loader(tmp_path, knowledge_service=None)
        loader.load_all(tmp_path)  # must not raise

    def test_knowledge_tool_without_service_logged_as_warning(
        self, tmp_path: Path, capsys
    ) -> None:
        _make_workflow_dir(tmp_path, "wf", tools_yaml=_KNOWLEDGE_TOOL_YAML)
        loader, _, _, _ = _make_loader(tmp_path, knowledge_service=None)
        loader.load_all(tmp_path)
        captured = capsys.readouterr()
        assert "wf" in captured.err

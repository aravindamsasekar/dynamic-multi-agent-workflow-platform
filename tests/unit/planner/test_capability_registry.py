"""Unit tests for CapabilityRegistry and capability descriptor models."""

from __future__ import annotations

import pytest

from platform.planner.capability_registry import CapabilityRegistry, DuplicateCapabilityError
from platform.planner.models import (
    AgentCapabilityDescriptor,
    OperationType,
    PatternCapabilityDescriptor,
    ToolCapabilityDescriptor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent(
    agent_id: str = "test_agent",
    capabilities: list[str] | None = None,
    task_types: list[str] | None = None,
) -> AgentCapabilityDescriptor:
    return AgentCapabilityDescriptor(
        agent_id=agent_id,
        name=agent_id.replace("_", " ").title(),
        description="A test agent.",
        capabilities=capabilities or ["do_thing"],
        supported_task_types=task_types or ["code_review"],
    )


def _tool(
    tool_name: str = "test_tool",
    capabilities: list[str] | None = None,
    operation_type: OperationType = OperationType.READ,
) -> ToolCapabilityDescriptor:
    return ToolCapabilityDescriptor(
        tool_name=tool_name,
        name=tool_name.replace("_", " ").title(),
        description="A test tool.",
        capabilities=capabilities or ["read_thing"],
        operation_type=operation_type,
        data_source="custom",
    )


def _pattern(
    pattern: str = "parallel_specialist",
    task_types: list[str] | None = None,
) -> PatternCapabilityDescriptor:
    return PatternCapabilityDescriptor(
        pattern=pattern,
        name=pattern.replace("_", " ").title(),
        description="A test pattern.",
        best_for=["analysis"],
        supported_task_types=task_types or ["code_review"],
    )


# ---------------------------------------------------------------------------
# Descriptor models
# ---------------------------------------------------------------------------


class TestAgentCapabilityDescriptor:
    def test_required_fields(self):
        d = _agent("my_agent", ["cap_a", "cap_b"], ["code_review"])
        assert d.agent_id == "my_agent"
        assert d.capabilities == ["cap_a", "cap_b"]
        assert d.supported_task_types == ["code_review"]

    def test_optional_fields_default_to_empty(self):
        d = AgentCapabilityDescriptor(
            agent_id="a",
            name="A",
            description="desc",
            capabilities=["c"],
            supported_task_types=["t"],
        )
        assert d.input_description == ""
        assert d.output_description == ""
        assert d.required_tool_capabilities == []

    def test_required_tool_capabilities_are_independent_between_instances(self):
        d1 = AgentCapabilityDescriptor(
            agent_id="a1", name="A1", description="", capabilities=[], supported_task_types=[]
        )
        d2 = AgentCapabilityDescriptor(
            agent_id="a2", name="A2", description="", capabilities=[], supported_task_types=[]
        )
        d1.required_tool_capabilities.append("tool_x")
        assert d2.required_tool_capabilities == []


class TestToolCapabilityDescriptor:
    def test_required_fields(self):
        d = _tool("github_get_pr", ["read_github_pr"])
        assert d.tool_name == "github_get_pr"
        assert "read_github_pr" in d.capabilities
        assert d.operation_type == OperationType.READ

    def test_safety_defaults(self):
        d = _tool()
        assert d.requires_credentials is False
        assert d.requires_mcp is False
        assert d.is_destructive is False
        assert d.requires_hitl is False

    def test_destructive_write_tool(self):
        d = ToolCapabilityDescriptor(
            tool_name="delete_branch",
            name="Delete Branch",
            description="Deletes a git branch.",
            capabilities=["delete_github_branch"],
            operation_type=OperationType.WRITE,
            data_source="github",
            is_destructive=True,
            requires_hitl=True,
        )
        assert d.is_destructive is True
        assert d.requires_hitl is True


class TestPatternCapabilityDescriptor:
    def test_required_fields(self):
        d = _pattern("parallel_specialist", ["code_review"])
        assert d.pattern == "parallel_specialist"
        assert "code_review" in d.supported_task_types

    def test_iteration_defaults_to_false(self):
        d = _pattern()
        assert d.supports_iteration is False

    def test_agent_count_bounds(self):
        d = PatternCapabilityDescriptor(
            pattern="peo",
            name="PEO",
            description="",
            best_for=[],
            supported_task_types=["research"],
            min_agents=3,
            max_agents=3,
        )
        assert d.min_agents == 3
        assert d.max_agents == 3


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestCapabilityRegistryRegistration:
    def test_register_and_get_agent(self):
        registry = CapabilityRegistry()
        descriptor = _agent("agent_a")
        registry.register_agent(descriptor)
        assert registry.get_agent("agent_a") is descriptor

    def test_register_and_get_tool(self):
        registry = CapabilityRegistry()
        descriptor = _tool("tool_a")
        registry.register_tool(descriptor)
        assert registry.get_tool("tool_a") is descriptor

    def test_register_and_get_pattern(self):
        registry = CapabilityRegistry()
        descriptor = _pattern("parallel_specialist")
        registry.register_pattern(descriptor)
        assert registry.get_pattern("parallel_specialist") is descriptor

    def test_get_unknown_agent_returns_none(self):
        registry = CapabilityRegistry()
        assert registry.get_agent("nonexistent") is None

    def test_get_unknown_tool_returns_none(self):
        registry = CapabilityRegistry()
        assert registry.get_tool("nonexistent") is None

    def test_get_unknown_pattern_returns_none(self):
        registry = CapabilityRegistry()
        assert registry.get_pattern("nonexistent") is None

    def test_duplicate_agent_raises(self):
        registry = CapabilityRegistry()
        registry.register_agent(_agent("dup"))
        with pytest.raises(DuplicateCapabilityError, match="dup"):
            registry.register_agent(_agent("dup"))

    def test_duplicate_tool_raises(self):
        registry = CapabilityRegistry()
        registry.register_tool(_tool("dup_tool"))
        with pytest.raises(DuplicateCapabilityError, match="dup_tool"):
            registry.register_tool(_tool("dup_tool"))

    def test_duplicate_pattern_raises(self):
        registry = CapabilityRegistry()
        registry.register_pattern(_pattern("router"))
        with pytest.raises(DuplicateCapabilityError, match="router"):
            registry.register_pattern(_pattern("router"))


# ---------------------------------------------------------------------------
# Capability tag queries
# ---------------------------------------------------------------------------


class TestCapabilityTagQueries:
    def test_find_agents_by_capability(self):
        registry = CapabilityRegistry()
        registry.register_agent(_agent("a1", capabilities=["fetch_data", "summarize"]))
        registry.register_agent(_agent("a2", capabilities=["summarize"]))
        registry.register_agent(_agent("a3", capabilities=["review_code"]))

        results = registry.find_agents_by_capability("summarize")
        ids = {d.agent_id for d in results}
        assert ids == {"a1", "a2"}

    def test_find_agents_by_capability_returns_empty_for_unknown(self):
        registry = CapabilityRegistry()
        registry.register_agent(_agent("a1", capabilities=["do_thing"]))
        assert registry.find_agents_by_capability("nonexistent_cap") == []

    def test_find_tools_by_capability(self):
        registry = CapabilityRegistry()
        registry.register_tool(_tool("t1", capabilities=["read_diff", "fetch_files"]))
        registry.register_tool(_tool("t2", capabilities=["fetch_files"]))
        registry.register_tool(_tool("t3", capabilities=["search_kb"]))

        results = registry.find_tools_by_capability("fetch_files")
        names = {d.tool_name for d in results}
        assert names == {"t1", "t2"}

    def test_find_tools_by_capability_returns_empty_for_unknown(self):
        registry = CapabilityRegistry()
        registry.register_tool(_tool("t1", capabilities=["read_thing"]))
        assert registry.find_tools_by_capability("phantom") == []

    def test_find_agents_by_task_type(self):
        registry = CapabilityRegistry()
        registry.register_agent(_agent("cr_agent", task_types=["code_review"]))
        registry.register_agent(_agent("support_agent", task_types=["support"]))

        results = registry.find_agents_by_task_type("code_review")
        assert len(results) == 1
        assert results[0].agent_id == "cr_agent"

    def test_find_agents_by_task_type_empty_for_unknown(self):
        registry = CapabilityRegistry()
        registry.register_agent(_agent("a", task_types=["code_review"]))
        assert registry.find_agents_by_task_type("unknown_type") == []


# ---------------------------------------------------------------------------
# Pattern task-type queries
# ---------------------------------------------------------------------------


class TestPatternTaskTypeQueries:
    def test_find_patterns_for_task_type(self):
        registry = CapabilityRegistry()
        registry.register_pattern(_pattern("parallel_specialist", task_types=["code_review"]))
        registry.register_pattern(_pattern("router", task_types=["support"]))

        results = registry.find_patterns_for_task_type("code_review")
        assert len(results) == 1
        assert results[0].pattern == "parallel_specialist"

    def test_find_patterns_returns_empty_for_unknown_task_type(self):
        registry = CapabilityRegistry()
        registry.register_pattern(_pattern("parallel_specialist", task_types=["code_review"]))
        assert registry.find_patterns_for_task_type("data_analysis") == []

    def test_get_default_pattern_for_task_type(self):
        registry = CapabilityRegistry()
        registry.register_pattern(_pattern("parallel_specialist", task_types=["code_review"]))
        result = registry.get_default_pattern_for_task_type("code_review")
        assert result is not None
        assert result.pattern == "parallel_specialist"

    def test_get_default_pattern_returns_none_for_unknown(self):
        registry = CapabilityRegistry()
        assert registry.get_default_pattern_for_task_type("nonexistent") is None


# ---------------------------------------------------------------------------
# PR review built-in registry
# ---------------------------------------------------------------------------


class TestPRReviewRegistry:
    @pytest.fixture
    def registry(self) -> CapabilityRegistry:
        return CapabilityRegistry.build_pr_review_registry()

    def test_all_four_pr_review_agents_present(self, registry: CapabilityRegistry):
        expected = {"pr_data_agent", "review_specialist", "risk_specialist", "synthesis_agent"}
        for agent_id in expected:
            assert registry.get_agent(agent_id) is not None, f"Missing agent: {agent_id}"

    def test_exactly_four_pr_review_agents(self, registry: CapabilityRegistry):
        agents = registry.find_agents_by_task_type("code_review")
        assert len(agents) == 4

    def test_all_five_pr_review_tools_present(self, registry: CapabilityRegistry):
        expected = {
            "github_get_pr",
            "github_get_files",
            "github_get_diff",
            "knowledge_search",
            "mcp_get_pr_comments",
        }
        for tool_name in expected:
            assert registry.get_tool(tool_name) is not None, f"Missing tool: {tool_name}"

    def test_code_review_maps_to_parallel_specialist(self, registry: CapabilityRegistry):
        pattern = registry.get_default_pattern_for_task_type("code_review")
        assert pattern is not None
        assert pattern.pattern == "parallel_specialist"

    def test_parallel_specialist_requires_reviewer(self, registry: CapabilityRegistry):
        pattern = registry.get_pattern("parallel_specialist")
        assert pattern is not None
        assert pattern.requires_reviewer is True

    def test_pr_data_agent_capabilities(self, registry: CapabilityRegistry):
        agent = registry.get_agent("pr_data_agent")
        assert agent is not None
        assert "fetch_pr_data" in agent.capabilities
        assert "fetch_github_diff" in agent.capabilities
        assert "fetch_changed_files" in agent.capabilities

    def test_synthesis_agent_has_synthesize_capability(self, registry: CapabilityRegistry):
        agent = registry.get_agent("synthesis_agent")
        assert agent is not None
        assert "synthesize_findings" in agent.capabilities

    def test_mcp_tool_has_mcp_flag(self, registry: CapabilityRegistry):
        tool = registry.get_tool("mcp_get_pr_comments")
        assert tool is not None
        assert tool.requires_mcp is True
        assert tool.operation_type == OperationType.READ

    def test_github_tools_are_read_only(self, registry: CapabilityRegistry):
        for tool_name in ("github_get_pr", "github_get_files", "github_get_diff"):
            tool = registry.get_tool(tool_name)
            assert tool is not None
            assert tool.operation_type == OperationType.READ
            assert tool.is_destructive is False

    def test_knowledge_search_is_search_type(self, registry: CapabilityRegistry):
        tool = registry.get_tool("knowledge_search")
        assert tool is not None
        assert tool.operation_type == OperationType.SEARCH

    def test_find_agents_by_capability_fetch_pr_data(self, registry: CapabilityRegistry):
        agents = registry.find_agents_by_capability("fetch_pr_data")
        assert len(agents) == 1
        assert agents[0].agent_id == "pr_data_agent"

    def test_find_agents_by_capability_assess_security(self, registry: CapabilityRegistry):
        agents = registry.find_agents_by_capability("assess_security")
        assert len(agents) == 1
        assert agents[0].agent_id == "risk_specialist"

    def test_find_tools_by_capability_search_knowledge(self, registry: CapabilityRegistry):
        tools = registry.find_tools_by_capability("search_knowledge")
        assert len(tools) == 1
        assert tools[0].tool_name == "knowledge_search"

    def test_all_three_patterns_registered(self, registry: CapabilityRegistry):
        for pattern in ("parallel_specialist", "router", "planner_executor_observer"):
            assert registry.get_pattern(pattern) is not None, f"Missing pattern: {pattern}"

    def test_router_supports_support_task_type(self, registry: CapabilityRegistry):
        patterns = registry.find_patterns_for_task_type("support")
        assert any(p.pattern == "router" for p in patterns)

    def test_peo_supports_research_task_type(self, registry: CapabilityRegistry):
        patterns = registry.find_patterns_for_task_type("research")
        assert any(p.pattern == "planner_executor_observer" for p in patterns)


# ---------------------------------------------------------------------------
# Prompt summary
# ---------------------------------------------------------------------------


class TestPromptSummary:
    def test_summary_contains_agent_section(self):
        registry = CapabilityRegistry()
        registry.register_agent(_agent("my_agent", capabilities=["do_thing"]))
        summary = registry.to_prompt_summary()
        assert "=== Registered Agents ===" in summary
        assert "my_agent" in summary
        assert "do_thing" in summary

    def test_summary_contains_tool_section(self):
        registry = CapabilityRegistry()
        registry.register_tool(_tool("my_tool", capabilities=["read_stuff"]))
        summary = registry.to_prompt_summary()
        assert "=== Registered Tools ===" in summary
        assert "my_tool" in summary
        assert "read_stuff" in summary

    def test_summary_contains_pattern_section(self):
        registry = CapabilityRegistry()
        registry.register_pattern(_pattern("parallel_specialist"))
        summary = registry.to_prompt_summary()
        assert "=== Execution Patterns ===" in summary
        assert "parallel_specialist" in summary

    def test_pr_review_registry_summary_contains_all_agents(self):
        registry = CapabilityRegistry.build_pr_review_registry()
        summary = registry.to_prompt_summary()
        for agent_id in ("pr_data_agent", "review_specialist", "risk_specialist", "synthesis_agent"):
            assert agent_id in summary

    def test_pr_review_registry_summary_contains_all_tools(self):
        registry = CapabilityRegistry.build_pr_review_registry()
        summary = registry.to_prompt_summary()
        for tool_name in (
            "github_get_pr",
            "github_get_files",
            "github_get_diff",
            "knowledge_search",
            "mcp_get_pr_comments",
        ):
            assert tool_name in summary

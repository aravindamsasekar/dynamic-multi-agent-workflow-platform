"""Unit tests for ToolSelector — deterministic, two-path logic."""

from __future__ import annotations

import pytest

from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.models import (
    AgentCapabilityDescriptor,
    GoalAnalysis,
    RiskLevel,
    RuntimeAgentDefinition,
    ToolCapabilityDescriptor,
    OperationType,
)
from platform.planner.tool_selector import ToolSelector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _analysis() -> GoalAnalysis:
    return GoalAnalysis(
        required_capabilities=["fetch_pr_data"],
        risk_level=RiskLevel.LOW,
        confidence=0.9,
        reasoning="Test.",
        constraints=[],
        requires_hitl=False,
    )


def _make_static(agent_id: str, capabilities: list[str]) -> RuntimeAgentDefinition:
    return RuntimeAgentDefinition(
        id=agent_id,
        name=agent_id,
        description="",
        capabilities=capabilities,
        tool_names=[],
        system_prompt="",
        generated=False,
    )


def _make_generated(
    agent_id: str,
    capabilities: list[str],
    tool_names: list[str],
) -> RuntimeAgentDefinition:
    return RuntimeAgentDefinition(
        id=agent_id,
        name=agent_id,
        description="",
        capabilities=capabilities,
        tool_names=tool_names,
        system_prompt="generated",
        generated=True,
    )


@pytest.fixture
def selector() -> ToolSelector:
    return ToolSelector()


@pytest.fixture
def registry() -> CapabilityRegistry:
    return CapabilityRegistry.build_pr_review_registry()


# ---------------------------------------------------------------------------
# Empty guard
# ---------------------------------------------------------------------------


class TestToolSelectorEmpty:
    def test_empty_agents_returns_empty_list(self, selector: ToolSelector, registry: CapabilityRegistry):
        result = selector.select(_analysis(), [], registry)
        assert result == []

    def test_select_is_synchronous(self, selector: ToolSelector):
        import inspect
        assert not inspect.iscoroutinefunction(selector.select)


# ---------------------------------------------------------------------------
# Static agent path
# ---------------------------------------------------------------------------


class TestToolSelectorStaticAgents:
    def test_static_agent_tools_selected_via_registry(
        self, selector: ToolSelector, registry: CapabilityRegistry
    ):
        agents = [_make_static("pr_data_agent", ["fetch_pr_data"])]
        result = selector.select(_analysis(), agents, registry)
        # pr_data_agent has required_tool_capabilities: read_github_pr, read_github_files, read_github_diff
        assert "github_get_pr" in result
        assert "github_get_files" in result
        assert "github_get_diff" in result

    def test_static_agent_unknown_id_is_silently_skipped(
        self, selector: ToolSelector, registry: CapabilityRegistry
    ):
        agents = [_make_static("ghost_agent", ["some_cap"])]
        result = selector.select(_analysis(), agents, registry)
        assert result == []

    def test_pr_review_all_static_agents_selects_five_tools(
        self, selector: ToolSelector, registry: CapabilityRegistry
    ):
        agents = [
            _make_static("pr_data_agent", ["fetch_pr_data"]),
            _make_static("review_specialist", ["review_code_quality"]),
            _make_static("risk_specialist", ["assess_security"]),
            _make_static("synthesis_agent", ["synthesize_findings"]),
        ]
        result = selector.select(_analysis(), agents, registry)
        expected = {
            "github_get_pr",
            "github_get_files",
            "github_get_diff",
            "knowledge_search",
            "mcp_get_pr_comments",
        }
        assert set(result) == expected

    def test_static_agent_tools_deduplicated_across_agents(
        self, selector: ToolSelector, registry: CapabilityRegistry
    ):
        # review_specialist and risk_specialist both need github_get_diff and knowledge_search
        agents = [
            _make_static("review_specialist", ["review_code_quality"]),
            _make_static("risk_specialist", ["assess_security"]),
        ]
        result = selector.select(_analysis(), agents, registry)
        assert result.count("github_get_diff") == 1
        assert result.count("knowledge_search") == 1


# ---------------------------------------------------------------------------
# Generated agent path
# ---------------------------------------------------------------------------


class TestToolSelectorGeneratedAgents:
    def test_generated_agent_tools_from_tool_names(
        self, selector: ToolSelector, registry: CapabilityRegistry
    ):
        agents = [_make_generated("cap_agent", ["some_cap"], tool_names=["github_get_pr"])]
        result = selector.select(_analysis(), agents, registry)
        assert result == ["github_get_pr"]

    def test_generated_agent_with_no_tool_names_returns_empty(
        self, selector: ToolSelector, registry: CapabilityRegistry
    ):
        agents = [_make_generated("cap_agent", ["some_cap"], tool_names=[])]
        result = selector.select(_analysis(), agents, registry)
        assert result == []

    def test_generated_agent_tools_deduplicated(
        self, selector: ToolSelector, registry: CapabilityRegistry
    ):
        agents = [
            _make_generated("agent_a", ["cap_a"], tool_names=["github_get_pr", "knowledge_search"]),
            _make_generated("agent_b", ["cap_b"], tool_names=["knowledge_search", "github_get_diff"]),
        ]
        result = selector.select(_analysis(), agents, registry)
        assert result.count("knowledge_search") == 1
        assert set(result) == {"github_get_pr", "knowledge_search", "github_get_diff"}


# ---------------------------------------------------------------------------
# Mixed static + generated
# ---------------------------------------------------------------------------


class TestToolSelectorMixed:
    def test_mixed_static_and_generated_agents(
        self, selector: ToolSelector, registry: CapabilityRegistry
    ):
        agents = [
            _make_static("pr_data_agent", ["fetch_pr_data"]),
            _make_generated("extra_agent", ["extra_cap"], tool_names=["knowledge_search"]),
        ]
        result = selector.select(_analysis(), agents, registry)
        # pr_data_agent contributes: github_get_pr, github_get_files, github_get_diff
        # extra_agent contributes: knowledge_search (via tool_names)
        assert "github_get_pr" in result
        assert "knowledge_search" in result

    def test_mixed_deduplication_across_paths(
        self, selector: ToolSelector, registry: CapabilityRegistry
    ):
        # Both static and generated agent contribute github_get_pr — should appear once.
        agents = [
            _make_static("pr_data_agent", ["fetch_pr_data"]),
            _make_generated("extra_agent", ["extra_cap"], tool_names=["github_get_pr"]),
        ]
        result = selector.select(_analysis(), agents, registry)
        assert result.count("github_get_pr") == 1

    def test_selected_tools_order_static_before_generated(
        self, selector: ToolSelector, registry: CapabilityRegistry
    ):
        # Static agent processed first → its tools appear first.
        agents = [
            _make_static("pr_data_agent", ["fetch_pr_data"]),
            _make_generated("extra_agent", ["extra_cap"], tool_names=["mcp_get_pr_comments"]),
        ]
        result = selector.select(_analysis(), agents, registry)
        # github_get_pr (from static) should appear before mcp_get_pr_comments (from generated)
        pr_idx = result.index("github_get_pr")
        mcp_idx = result.index("mcp_get_pr_comments")
        assert pr_idx < mcp_idx

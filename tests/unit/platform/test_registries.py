"""Unit tests for WorkflowRegistry, AgentRegistry, and ToolRegistry."""

from __future__ import annotations

import pytest

from platform.core.exceptions import AgentNotFound, ToolNotFound, WorkflowNotFound
from platform.core.interfaces.tool import IToolAdapter
from platform.core.models.agent import AgentDefinition
from platform.core.models.tool import AdapterType, ToolCall, ToolDefinition, ToolResult
from platform.core.models.workflow import PatternType, WorkflowDefinition
from platform.registries.agent_registry import AgentRegistry
from platform.registries.tool_registry import ToolRegistry
from platform.registries.workflow_registry import WorkflowRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_workflow(workflow_id: str = "wf-1") -> WorkflowDefinition:
    return WorkflowDefinition(
        workflow_id=workflow_id,
        name="Test Workflow",
        pattern=PatternType.ROUTER,
    )


def make_agent(agent_id: str = "agent-1") -> AgentDefinition:
    return AgentDefinition(
        agent_id=agent_id,
        name="Test Agent",
        system_prompt="You are a test agent.",
    )


class _StubAdapter(IToolAdapter):
    async def execute(self, call: ToolCall) -> ToolResult:
        return ToolResult(tool_use_id=call.tool_use_id, content="ok")


# ---------------------------------------------------------------------------
# WorkflowRegistry
# ---------------------------------------------------------------------------


class TestWorkflowRegistry:
    def test_register_and_get(self):
        registry = WorkflowRegistry()
        wf = make_workflow("wf-1")
        registry.register(wf)
        assert registry.get("wf-1") is wf

    def test_get_raises_when_not_found(self):
        registry = WorkflowRegistry()
        with pytest.raises(WorkflowNotFound):
            registry.get("missing")

    def test_list_all_returns_all(self):
        registry = WorkflowRegistry()
        wf1 = make_workflow("wf-1")
        wf2 = make_workflow("wf-2")
        registry.register(wf1)
        registry.register(wf2)
        result = registry.list_all()
        assert len(result) == 2
        assert wf1 in result
        assert wf2 in result

    def test_list_all_empty(self):
        assert WorkflowRegistry().list_all() == []

    def test_exists_true(self):
        registry = WorkflowRegistry()
        registry.register(make_workflow("wf-1"))
        assert registry.exists("wf-1") is True

    def test_exists_false(self):
        assert WorkflowRegistry().exists("missing") is False

    def test_clear_empties_registry(self):
        registry = WorkflowRegistry()
        registry.register(make_workflow("wf-1"))
        registry.clear()
        assert registry.list_all() == []
        assert registry.exists("wf-1") is False

    def test_register_overwrites_existing(self):
        registry = WorkflowRegistry()
        wf_v1 = make_workflow("wf-1")
        wf_v2 = WorkflowDefinition(
            workflow_id="wf-1",
            name="Updated Workflow",
            pattern=PatternType.PARALLEL_SPECIALIST,
        )
        registry.register(wf_v1)
        registry.register(wf_v2)
        assert registry.get("wf-1").name == "Updated Workflow"
        assert len(registry.list_all()) == 1


# ---------------------------------------------------------------------------
# AgentRegistry
# ---------------------------------------------------------------------------


class TestAgentRegistry:
    def test_register_and_get(self):
        registry = AgentRegistry()
        agent = make_agent("agent-1")
        registry.register(agent)
        assert registry.get("agent-1") is agent

    def test_get_raises_when_not_found(self):
        registry = AgentRegistry()
        with pytest.raises(AgentNotFound):
            registry.get("missing")

    def test_list_all_returns_all(self):
        registry = AgentRegistry()
        a1 = make_agent("a-1")
        a2 = make_agent("a-2")
        registry.register(a1)
        registry.register(a2)
        result = registry.list_all()
        assert len(result) == 2
        assert a1 in result
        assert a2 in result

    def test_list_all_empty(self):
        assert AgentRegistry().list_all() == []

    def test_exists_true(self):
        registry = AgentRegistry()
        registry.register(make_agent("a-1"))
        assert registry.exists("a-1") is True

    def test_exists_false(self):
        assert AgentRegistry().exists("missing") is False

    def test_clear_empties_registry(self):
        registry = AgentRegistry()
        registry.register(make_agent("a-1"))
        registry.clear()
        assert registry.list_all() == []
        assert registry.exists("a-1") is False


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        adapter = _StubAdapter()
        registry.register("search", adapter)
        assert registry.get("search") is adapter

    def test_get_raises_when_not_found(self):
        registry = ToolRegistry()
        with pytest.raises(ToolNotFound):
            registry.get("missing")

    def test_list_all_returns_tool_names(self):
        registry = ToolRegistry()
        registry.register("search", _StubAdapter())
        registry.register("email", _StubAdapter())
        names = registry.list_all()
        assert sorted(names) == ["email", "search"]

    def test_list_all_empty(self):
        assert ToolRegistry().list_all() == []

    def test_exists_true(self):
        registry = ToolRegistry()
        registry.register("search", _StubAdapter())
        assert registry.exists("search") is True

    def test_exists_false(self):
        assert ToolRegistry().exists("missing") is False

    def test_clear_empties_registry(self):
        registry = ToolRegistry()
        registry.register("search", _StubAdapter())
        registry.clear()
        assert registry.list_all() == []
        assert registry.exists("search") is False

    def test_register_overwrites_existing(self):
        registry = ToolRegistry()
        a1 = _StubAdapter()
        a2 = _StubAdapter()
        registry.register("search", a1)
        registry.register("search", a2)
        assert registry.get("search") is a2
        assert len(registry.list_all()) == 1

"""Unit tests for AgentRegistry — including Phase C unregister method."""

from __future__ import annotations

import pytest

from platform.core.models.agent import AgentDefinition, LLMConfig
from platform.registries.agent_registry import AgentRegistry


def _make_agent(agent_id: str, name: str = "") -> AgentDefinition:
    return AgentDefinition(
        agent_id=agent_id,
        name=name or agent_id,
        description="",
        system_prompt=f"You are {agent_id}.",
        tool_names=[],
        llm_config=LLMConfig(),
    )


@pytest.fixture
def registry() -> AgentRegistry:
    return AgentRegistry()


# ---------------------------------------------------------------------------
# Basic register / get / exists
# ---------------------------------------------------------------------------


class TestAgentRegistryBasic:
    def test_register_and_get(self, registry):
        registry.register(_make_agent("agent_a"))
        result = registry.get("agent_a")
        assert result.agent_id == "agent_a"

    def test_exists_returns_true_after_register(self, registry):
        registry.register(_make_agent("agent_b"))
        assert registry.exists("agent_b") is True

    def test_exists_returns_false_for_unknown(self, registry):
        assert registry.exists("ghost") is False

    def test_list_all_returns_all_registered(self, registry):
        registry.register(_make_agent("a"))
        registry.register(_make_agent("b"))
        ids = {a.agent_id for a in registry.list_all()}
        assert ids == {"a", "b"}


# ---------------------------------------------------------------------------
# unregister (Phase C)
# ---------------------------------------------------------------------------


class TestAgentRegistryUnregister:
    def test_unregister_removes_agent(self, registry):
        registry.register(_make_agent("gen_plan_abc_my_cap"))
        registry.unregister("gen_plan_abc_my_cap")
        assert not registry.exists("gen_plan_abc_my_cap")

    def test_unregister_nonexistent_is_noop(self, registry):
        # Must not raise
        registry.unregister("does_not_exist")

    def test_unregister_does_not_affect_other_agents(self, registry):
        registry.register(_make_agent("keep_me"))
        registry.register(_make_agent("remove_me"))
        registry.unregister("remove_me")
        assert registry.exists("keep_me")
        assert not registry.exists("remove_me")

    def test_unregister_after_clear_is_noop(self, registry):
        registry.register(_make_agent("gen_x"))
        registry.clear()
        registry.unregister("gen_x")  # must not raise

    def test_double_unregister_is_noop(self, registry):
        registry.register(_make_agent("gen_y"))
        registry.unregister("gen_y")
        registry.unregister("gen_y")  # second call must not raise
        assert not registry.exists("gen_y")

    def test_unregister_allows_re_register_with_same_id(self, registry):
        registry.register(_make_agent("gen_z", name="original"))
        registry.unregister("gen_z")
        registry.register(_make_agent("gen_z", name="new"))
        assert registry.get("gen_z").name == "new"

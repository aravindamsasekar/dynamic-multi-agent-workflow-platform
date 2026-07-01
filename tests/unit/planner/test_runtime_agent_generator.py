"""Unit tests for RuntimeAgentDefinition and RuntimeAgentGenerator."""

from __future__ import annotations

import pytest

from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.models import (
    AgentCapabilityDescriptor,
    OperationType,
    RuntimeAgentDefinition,
    ToolCapabilityDescriptor,
)
from platform.planner.runtime_agent_generator import RuntimeAgentGenerator

# Stable plan_id used across all generator tests.
_PLAN_ID = "test-plan-abc-123"


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def _pr_registry() -> CapabilityRegistry:
    """Full PR review registry (4 agents, 5 tools, 3 patterns)."""
    return CapabilityRegistry.build_pr_review_registry()


def _empty_registry() -> CapabilityRegistry:
    return CapabilityRegistry()


def _registry_with_tool_for_cap(capability: str) -> CapabilityRegistry:
    """Registry with no agents but one tool that exposes `capability`."""
    reg = CapabilityRegistry()
    reg.register_tool(ToolCapabilityDescriptor(
        tool_name=f"{capability}_tool",
        name=f"{capability} Tool",
        description=f"Tool for {capability}.",
        capabilities=[capability],
        operation_type=OperationType.READ,
        data_source="custom",
    ))
    return reg


def _registry_with_two_tools_for_cap(capability: str) -> CapabilityRegistry:
    reg = CapabilityRegistry()
    reg.register_tool(ToolCapabilityDescriptor(
        tool_name=f"{capability}_tool_a",
        name="Tool A",
        description="",
        capabilities=[capability],
        operation_type=OperationType.READ,
        data_source="custom",
    ))
    reg.register_tool(ToolCapabilityDescriptor(
        tool_name=f"{capability}_tool_b",
        name="Tool B",
        description="",
        capabilities=[capability],
        operation_type=OperationType.READ,
        data_source="custom",
    ))
    return reg


def _registry_with_agent(agent_id: str, capabilities: list[str]) -> CapabilityRegistry:
    reg = CapabilityRegistry()
    reg.register_agent(AgentCapabilityDescriptor(
        agent_id=agent_id,
        name=f"{agent_id} Name",
        description=f"{agent_id} description.",
        capabilities=capabilities,
    ))
    return reg


# ---------------------------------------------------------------------------
# RuntimeAgentDefinition — model tests
# ---------------------------------------------------------------------------


class TestRuntimeAgentDefinition:
    def test_all_fields_stored(self):
        agent = RuntimeAgentDefinition(
            id="test_agent",
            name="Test Agent",
            description="A test agent.",
            capabilities=["cap_a"],
            tool_names=["tool_x"],
            system_prompt="Do something.",
            generated=True,
        )
        assert agent.id == "test_agent"
        assert agent.name == "Test Agent"
        assert agent.description == "A test agent."
        assert agent.capabilities == ["cap_a"]
        assert agent.tool_names == ["tool_x"]
        assert agent.system_prompt == "Do something."
        assert agent.generated is True

    def test_generated_flag_distinguishes_static(self):
        static = RuntimeAgentDefinition(
            id="s", name="S", description="", capabilities=[], tool_names=[],
            system_prompt="", generated=False,
        )
        dynamic = RuntimeAgentDefinition(
            id="d", name="D", description="", capabilities=[], tool_names=[],
            system_prompt="", generated=True,
        )
        assert static.generated is False
        assert dynamic.generated is True

    def test_to_dict_has_all_keys(self):
        agent = RuntimeAgentDefinition(
            id="a", name="A", description="desc", capabilities=["c1"],
            tool_names=["t1"], system_prompt="prompt", generated=True,
        )
        d = agent.to_dict()
        assert set(d.keys()) == {"id", "name", "description", "capabilities",
                                  "tool_names", "system_prompt", "generated"}

    def test_to_dict_values_match(self):
        agent = RuntimeAgentDefinition(
            id="agent_x", name="Agent X", description="does X",
            capabilities=["cap_x", "cap_y"], tool_names=["tool_1", "tool_2"],
            system_prompt="Execute cap_x.", generated=True,
        )
        d = agent.to_dict()
        assert d["id"] == "agent_x"
        assert d["capabilities"] == ["cap_x", "cap_y"]
        assert d["tool_names"] == ["tool_1", "tool_2"]
        assert d["generated"] is True

    def test_from_dict_round_trip(self):
        agent = RuntimeAgentDefinition(
            id="round_trip_agent",
            name="Round Trip Agent",
            description="Serialisation test.",
            capabilities=["cap_a", "cap_b"],
            tool_names=["tool_a"],
            system_prompt="System prompt text.",
            generated=False,
        )
        restored = RuntimeAgentDefinition.from_dict(agent.to_dict())
        assert restored.id == agent.id
        assert restored.name == agent.name
        assert restored.description == agent.description
        assert restored.capabilities == agent.capabilities
        assert restored.tool_names == agent.tool_names
        assert restored.system_prompt == agent.system_prompt
        assert restored.generated == agent.generated

    def test_from_dict_generated_flag_preserved(self):
        for flag in (True, False):
            d = RuntimeAgentDefinition(
                id="x", name="X", description="", capabilities=[],
                tool_names=[], system_prompt="", generated=flag,
            ).to_dict()
            restored = RuntimeAgentDefinition.from_dict(d)
            assert restored.generated is flag

    def test_to_dict_returns_copy_of_lists(self):
        agent = RuntimeAgentDefinition(
            id="a", name="A", description="", capabilities=["c"],
            tool_names=["t"], system_prompt="", generated=True,
        )
        d = agent.to_dict()
        d["capabilities"].append("mutated")
        assert agent.capabilities == ["c"]

    def test_empty_lists_serialise_cleanly(self):
        agent = RuntimeAgentDefinition(
            id="a", name="A", description="", capabilities=[],
            tool_names=[], system_prompt="", generated=True,
        )
        restored = RuntimeAgentDefinition.from_dict(agent.to_dict())
        assert restored.capabilities == []
        assert restored.tool_names == []


# ---------------------------------------------------------------------------
# Static agent (generated=False)
# ---------------------------------------------------------------------------


class TestRuntimeAgentGeneratorStaticAgent:
    def test_known_cap_returns_one_entry(self):
        gen = RuntimeAgentGenerator(_pr_registry())
        result = gen.generate(["fetch_pr_data"], plan_id=_PLAN_ID)
        assert len(result) == 1

    def test_known_cap_generated_false(self):
        gen = RuntimeAgentGenerator(_pr_registry())
        result = gen.generate(["fetch_pr_data"], plan_id=_PLAN_ID)
        assert result[0].generated is False

    def test_known_cap_preserves_agent_id(self):
        gen = RuntimeAgentGenerator(_pr_registry())
        result = gen.generate(["fetch_pr_data"], plan_id=_PLAN_ID)
        assert result[0].id == "pr_data_agent"

    def test_known_cap_preserves_name(self):
        gen = RuntimeAgentGenerator(_pr_registry())
        result = gen.generate(["fetch_pr_data"], plan_id=_PLAN_ID)
        assert result[0].name == "PR Data Agent"

    def test_known_cap_preserves_description(self):
        gen = RuntimeAgentGenerator(_pr_registry())
        result = gen.generate(["fetch_pr_data"], plan_id=_PLAN_ID)
        assert "pull request" in result[0].description.lower()

    def test_known_cap_preserves_all_capabilities(self):
        gen = RuntimeAgentGenerator(_pr_registry())
        result = gen.generate(["fetch_pr_data"], plan_id=_PLAN_ID)
        assert "fetch_pr_data" in result[0].capabilities
        assert "fetch_github_diff" in result[0].capabilities
        assert "fetch_changed_files" in result[0].capabilities

    def test_known_cap_static_agent_system_prompt_empty(self):
        gen = RuntimeAgentGenerator(_pr_registry())
        result = gen.generate(["fetch_pr_data"], plan_id=_PLAN_ID)
        assert result[0].system_prompt == ""

    def test_known_cap_static_agent_tool_names_empty(self):
        # Static agents' tools are managed by the existing ToolSelector pipeline.
        gen = RuntimeAgentGenerator(_pr_registry())
        result = gen.generate(["fetch_pr_data"], plan_id=_PLAN_ID)
        assert result[0].tool_names == []


# ---------------------------------------------------------------------------
# Generated agent (generated=True)
# ---------------------------------------------------------------------------


class TestRuntimeAgentGeneratorGenerated:
    def test_unknown_cap_returns_one_entry(self):
        gen = RuntimeAgentGenerator(_empty_registry())
        result = gen.generate(["filesystem_read"], plan_id=_PLAN_ID)
        assert len(result) == 1

    def test_unknown_cap_generated_true(self):
        gen = RuntimeAgentGenerator(_empty_registry())
        result = gen.generate(["filesystem_read"], plan_id=_PLAN_ID)
        assert result[0].generated is True

    def test_unknown_cap_agent_id_format(self):
        # Generated IDs are scoped to the plan: gen_{plan_id}_{capability}
        gen = RuntimeAgentGenerator(_empty_registry())
        result = gen.generate(["filesystem_read"], plan_id=_PLAN_ID)
        assert result[0].id == f"gen_{_PLAN_ID}_filesystem_read"

    def test_unknown_cap_id_starts_with_gen_prefix(self):
        gen = RuntimeAgentGenerator(_empty_registry())
        result = gen.generate(["filesystem_read"], plan_id=_PLAN_ID)
        assert result[0].id.startswith("gen_")

    def test_unknown_cap_id_contains_capability(self):
        gen = RuntimeAgentGenerator(_empty_registry())
        result = gen.generate(["filesystem_read"], plan_id=_PLAN_ID)
        assert "filesystem_read" in result[0].id

    def test_unknown_cap_id_contains_plan_id(self):
        gen = RuntimeAgentGenerator(_empty_registry())
        result = gen.generate(["filesystem_read"], plan_id=_PLAN_ID)
        assert _PLAN_ID in result[0].id

    def test_unknown_cap_name_multi_word(self):
        gen = RuntimeAgentGenerator(_empty_registry())
        result = gen.generate(["filesystem_read"], plan_id=_PLAN_ID)
        assert result[0].name == "Filesystem Read Agent"

    def test_unknown_cap_name_single_word(self):
        gen = RuntimeAgentGenerator(_empty_registry())
        result = gen.generate(["deploy"], plan_id=_PLAN_ID)
        assert result[0].name == "Deploy Agent"

    def test_unknown_cap_name_three_words(self):
        gen = RuntimeAgentGenerator(_empty_registry())
        result = gen.generate(["send_slack_message"], plan_id=_PLAN_ID)
        assert result[0].name == "Send Slack Message Agent"

    def test_unknown_cap_description_template(self):
        gen = RuntimeAgentGenerator(_empty_registry())
        result = gen.generate(["filesystem_read"], plan_id=_PLAN_ID)
        assert result[0].description == "Agent responsible for the 'filesystem_read' capability."

    def test_unknown_cap_system_prompt_template(self):
        gen = RuntimeAgentGenerator(_empty_registry())
        result = gen.generate(["filesystem_read"], plan_id=_PLAN_ID)
        expected = (
            "You are responsible for executing the capability 'filesystem_read'. "
            "Use only the assigned tools. Return concise structured results."
        )
        assert result[0].system_prompt == expected

    def test_unknown_cap_system_prompt_contains_capability_name(self):
        gen = RuntimeAgentGenerator(_empty_registry())
        result = gen.generate(["knowledge_search"], plan_id=_PLAN_ID)
        assert "knowledge_search" in result[0].system_prompt

    def test_unknown_cap_capabilities_list_contains_only_that_cap(self):
        gen = RuntimeAgentGenerator(_empty_registry())
        result = gen.generate(["filesystem_read"], plan_id=_PLAN_ID)
        assert result[0].capabilities == ["filesystem_read"]

    def test_unknown_cap_no_tools_when_registry_has_no_matching_tool(self):
        gen = RuntimeAgentGenerator(_empty_registry())
        result = gen.generate(["filesystem_read"], plan_id=_PLAN_ID)
        assert result[0].tool_names == []

    def test_unknown_cap_no_tools_on_pr_registry_for_agent_caps(self):
        # PR registry tools have tool-level caps (read_github_pr etc.),
        # not agent-level caps — so no tool should match "fetch_pr_data".
        gen = RuntimeAgentGenerator(_pr_registry())
        result = gen.generate(["unknown_cap"], plan_id=_PLAN_ID)
        assert result[0].tool_names == []


# ---------------------------------------------------------------------------
# Tool assignment
# ---------------------------------------------------------------------------


class TestRuntimeAgentGeneratorToolAssignment:
    def test_generated_agent_gets_matching_tools(self):
        reg = _registry_with_tool_for_cap("filesystem_read")
        gen = RuntimeAgentGenerator(reg)
        result = gen.generate(["filesystem_read"], plan_id=_PLAN_ID)
        assert result[0].tool_names == ["filesystem_read_tool"]

    def test_generated_agent_gets_all_tools_for_capability(self):
        reg = _registry_with_two_tools_for_cap("filesystem_read")
        gen = RuntimeAgentGenerator(reg)
        result = gen.generate(["filesystem_read"], plan_id=_PLAN_ID)
        assert "filesystem_read_tool_a" in result[0].tool_names
        assert "filesystem_read_tool_b" in result[0].tool_names
        assert len(result[0].tool_names) == 2

    def test_no_duplicate_tools_assigned(self):
        # A tool exposes two capabilities; if both capabilities go to the same
        # generated agent (impossible — each cap gets its own agent), but test
        # that tool deduplication within a single capability works.
        reg = CapabilityRegistry()
        reg.register_tool(ToolCapabilityDescriptor(
            tool_name="shared_tool",
            name="Shared Tool",
            description="",
            capabilities=["cap_x", "cap_y"],
            operation_type=OperationType.READ,
            data_source="custom",
        ))
        gen = RuntimeAgentGenerator(reg)
        result = gen.generate(["cap_x"], plan_id=_PLAN_ID)
        # Only one tool registered — should appear exactly once
        assert result[0].tool_names.count("shared_tool") == 1

    def test_static_agent_has_no_tool_names(self):
        gen = RuntimeAgentGenerator(_pr_registry())
        result = gen.generate(["review_code_quality"], plan_id=_PLAN_ID)
        assert result[0].generated is False
        assert result[0].tool_names == []


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestRuntimeAgentGeneratorDeduplication:
    def test_duplicate_cap_in_input_yields_one_entry(self):
        gen = RuntimeAgentGenerator(_empty_registry())
        result = gen.generate(["filesystem_read", "filesystem_read"], plan_id=_PLAN_ID)
        assert len(result) == 1

    def test_duplicate_cap_preserves_first_occurrence(self):
        gen = RuntimeAgentGenerator(_empty_registry())
        result = gen.generate(["filesystem_read", "filesystem_read"], plan_id=_PLAN_ID)
        assert result[0].id == f"gen_{_PLAN_ID}_filesystem_read"

    def test_same_static_agent_for_two_caps_appears_once(self):
        # pr_data_agent covers both fetch_pr_data AND fetch_github_diff
        gen = RuntimeAgentGenerator(_pr_registry())
        result = gen.generate(["fetch_pr_data", "fetch_github_diff"], plan_id=_PLAN_ID)
        agent_ids = [r.id for r in result]
        assert agent_ids.count("pr_data_agent") == 1

    def test_same_static_agent_three_caps_appears_once(self):
        gen = RuntimeAgentGenerator(_pr_registry())
        result = gen.generate(["fetch_pr_data", "fetch_github_diff", "fetch_changed_files"], plan_id=_PLAN_ID)
        assert len(result) == 1
        assert result[0].id == "pr_data_agent"


# ---------------------------------------------------------------------------
# Multiple capabilities
# ---------------------------------------------------------------------------


class TestRuntimeAgentGeneratorMultiple:
    def test_two_different_known_caps_two_agents(self):
        gen = RuntimeAgentGenerator(_pr_registry())
        # fetch_pr_data → pr_data_agent; review_code_quality → review_specialist
        result = gen.generate(["fetch_pr_data", "review_code_quality"], plan_id=_PLAN_ID)
        assert len(result) == 2
        ids = {r.id for r in result}
        assert ids == {"pr_data_agent", "review_specialist"}

    def test_two_unknown_caps_two_generated_agents(self):
        gen = RuntimeAgentGenerator(_empty_registry())
        result = gen.generate(["filesystem_read", "knowledge_search"], plan_id=_PLAN_ID)
        assert len(result) == 2
        ids = {r.id for r in result}
        assert ids == {f"gen_{_PLAN_ID}_filesystem_read", f"gen_{_PLAN_ID}_knowledge_search"}

    def test_mixed_caps_correct_generated_flags(self):
        gen = RuntimeAgentGenerator(_pr_registry())
        result = gen.generate(["fetch_pr_data", "filesystem_read"], plan_id=_PLAN_ID)
        static_agents = [r for r in result if not r.generated]
        generated_agents = [r for r in result if r.generated]
        assert len(static_agents) == 1
        assert static_agents[0].id == "pr_data_agent"
        assert len(generated_agents) == 1
        assert generated_agents[0].id == f"gen_{_PLAN_ID}_filesystem_read"

    def test_order_preserved_known_first(self):
        gen = RuntimeAgentGenerator(_pr_registry())
        result = gen.generate(["fetch_pr_data", "filesystem_read"], plan_id=_PLAN_ID)
        assert result[0].id == "pr_data_agent"
        assert result[1].id == f"gen_{_PLAN_ID}_filesystem_read"

    def test_order_preserved_unknown_first(self):
        gen = RuntimeAgentGenerator(_pr_registry())
        result = gen.generate(["filesystem_read", "fetch_pr_data"], plan_id=_PLAN_ID)
        assert result[0].id == f"gen_{_PLAN_ID}_filesystem_read"
        assert result[1].id == "pr_data_agent"

    def test_all_four_pr_caps_returns_four_static_agents(self):
        gen = RuntimeAgentGenerator(_pr_registry())
        result = gen.generate([
            "fetch_pr_data", "review_code_quality", "assess_security", "synthesize_findings",
        ], plan_id=_PLAN_ID)
        assert len(result) == 4
        assert all(not r.generated for r in result)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestRuntimeAgentGeneratorEdgeCases:
    def test_empty_input_returns_empty_list(self):
        gen = RuntimeAgentGenerator(_pr_registry())
        result = gen.generate([], plan_id=_PLAN_ID)
        assert result == []

    def test_empty_input_empty_registry_returns_empty(self):
        gen = RuntimeAgentGenerator(_empty_registry())
        assert gen.generate([], plan_id=_PLAN_ID) == []

    def test_all_unknown_caps_on_empty_registry(self):
        gen = RuntimeAgentGenerator(_empty_registry())
        result = gen.generate(["cap_a", "cap_b", "cap_c"], plan_id=_PLAN_ID)
        assert len(result) == 3
        assert all(r.generated for r in result)

    def test_unknown_cap_on_pr_registry_generates_agent(self):
        gen = RuntimeAgentGenerator(_pr_registry())
        result = gen.generate(["make_coffee"], plan_id=_PLAN_ID)
        assert len(result) == 1
        assert result[0].generated is True
        assert result[0].id == f"gen_{_PLAN_ID}_make_coffee"

    def test_different_plan_ids_produce_different_agent_ids(self):
        gen = RuntimeAgentGenerator(_empty_registry())
        result_a = gen.generate(["filesystem_read"], plan_id="plan-aaa")
        result_b = gen.generate(["filesystem_read"], plan_id="plan-bbb")
        assert result_a[0].id != result_b[0].id
        assert "plan-aaa" in result_a[0].id
        assert "plan-bbb" in result_b[0].id

    def test_generated_id_cannot_collide_with_static_id(self):
        gen = RuntimeAgentGenerator(_pr_registry())
        result = gen.generate(["fetch_pr_data", "make_coffee"], plan_id=_PLAN_ID)
        static_ids = {r.id for r in result if not r.generated}
        generated_ids = {r.id for r in result if r.generated}
        assert static_ids.isdisjoint(generated_ids)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestRuntimeAgentGeneratorDeterminism:
    def test_same_input_produces_identical_output(self):
        caps = ["filesystem_read", "fetch_pr_data", "knowledge_search"]
        reg = _pr_registry()
        gen = RuntimeAgentGenerator(reg)
        result_a = gen.generate(caps, plan_id=_PLAN_ID)
        result_b = gen.generate(caps, plan_id=_PLAN_ID)
        assert [r.to_dict() for r in result_a] == [r.to_dict() for r in result_b]

    def test_generated_name_is_always_identical(self):
        gen = RuntimeAgentGenerator(_empty_registry())
        names = [gen.generate(["filesystem_read"], plan_id=_PLAN_ID)[0].name for _ in range(5)]
        assert len(set(names)) == 1

    def test_generated_prompt_is_always_identical(self):
        gen = RuntimeAgentGenerator(_empty_registry())
        prompts = [gen.generate(["filesystem_read"], plan_id=_PLAN_ID)[0].system_prompt for _ in range(5)]
        assert len(set(prompts)) == 1

    def test_generator_has_no_mutable_state_between_calls(self):
        reg = _pr_registry()
        gen = RuntimeAgentGenerator(reg)
        gen.generate(["fetch_pr_data", "filesystem_read"], plan_id=_PLAN_ID)
        # Second call must not be affected by first call's internal state
        result = gen.generate(["fetch_pr_data"], plan_id=_PLAN_ID)
        assert len(result) == 1
        assert result[0].id == "pr_data_agent"


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


class TestRuntimeAgentDefinitionSerialization:
    def test_generated_agent_survives_round_trip(self):
        gen = RuntimeAgentGenerator(_empty_registry())
        original = gen.generate(["filesystem_read"], plan_id=_PLAN_ID)[0]
        restored = RuntimeAgentDefinition.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.system_prompt == original.system_prompt
        assert restored.generated is True

    def test_static_agent_reference_survives_round_trip(self):
        gen = RuntimeAgentGenerator(_pr_registry())
        original = gen.generate(["fetch_pr_data"], plan_id=_PLAN_ID)[0]
        restored = RuntimeAgentDefinition.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.capabilities == original.capabilities
        assert restored.generated is False

    def test_list_of_agents_serialises_cleanly(self):
        gen = RuntimeAgentGenerator(_pr_registry())
        agents = gen.generate(["fetch_pr_data", "filesystem_read"], plan_id=_PLAN_ID)
        serialised = [a.to_dict() for a in agents]
        restored = [RuntimeAgentDefinition.from_dict(d) for d in serialised]
        for orig, rest in zip(agents, restored):
            assert orig.id == rest.id
            assert orig.generated == rest.generated

    def test_to_dict_is_json_serialisable(self):
        import json
        gen = RuntimeAgentGenerator(_empty_registry())
        agent = gen.generate(["filesystem_read"], plan_id=_PLAN_ID)[0]
        json_str = json.dumps(agent.to_dict())
        data = json.loads(json_str)
        restored = RuntimeAgentDefinition.from_dict(data)
        assert restored.id == agent.id

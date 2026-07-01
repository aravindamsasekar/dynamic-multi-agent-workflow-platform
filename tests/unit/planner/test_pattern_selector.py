"""Unit tests for PatternSelector — deterministic, capability-driven."""

from __future__ import annotations

import pytest

from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.models import (
    GoalAnalysis,
    PatternCapabilityDescriptor,
    RiskLevel,
    RuntimeAgentDefinition,
)
from platform.planner.pattern_selector import PatternSelector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _full_registry() -> CapabilityRegistry:
    """Registry containing all three patterns with their trigger_capabilities."""
    registry = CapabilityRegistry()
    registry.register_pattern(PatternCapabilityDescriptor(
        pattern="router",
        name="Router",
        description="Routes by classification.",
        best_for=[],
        supported_task_types=["support"],
        trigger_capabilities=["classify_intent"],
    ))
    registry.register_pattern(PatternCapabilityDescriptor(
        pattern="planner_executor_observer",
        name="PEO",
        description="Iterative research.",
        best_for=[],
        supported_task_types=["research"],
        trigger_capabilities=["iterative_research"],
    ))
    registry.register_pattern(PatternCapabilityDescriptor(
        pattern="parallel_specialist",
        name="Parallel Specialist",
        description="Parallel specialists.",
        best_for=[],
        supported_task_types=["code_review"],
        trigger_capabilities=[],
    ))
    return registry


def _make_agent(agent_id: str, capabilities: list[str], generated: bool = False) -> RuntimeAgentDefinition:
    return RuntimeAgentDefinition(
        id=agent_id,
        name=agent_id,
        description="",
        capabilities=capabilities,
        tool_names=[],
        system_prompt="",
        generated=generated,
    )


def _analysis() -> GoalAnalysis:
    return GoalAnalysis(
        required_capabilities=["fetch_pr_data"],
        risk_level=RiskLevel.LOW,
        confidence=0.9,
        reasoning="Test.",
        constraints=[],
        requires_hitl=False,
    )


@pytest.fixture
def selector() -> PatternSelector:
    return PatternSelector()


# ---------------------------------------------------------------------------
# Empty agents guard
# ---------------------------------------------------------------------------


class TestPatternSelectorEmptyAgents:
    def test_empty_agents_returns_empty_string(self, selector: PatternSelector):
        registry = _full_registry()
        result = selector.select(_analysis(), [], registry)
        assert result == ""

    def test_empty_agents_returns_empty_even_with_fallback_pattern(self, selector: PatternSelector):
        registry = CapabilityRegistry()
        registry.register_pattern(PatternCapabilityDescriptor(
            pattern="parallel_specialist",
            name="Parallel Specialist",
            description="",
            best_for=[],
            supported_task_types=["code_review"],
            trigger_capabilities=[],
        ))
        result = selector.select(_analysis(), [], registry)
        assert result == ""


# ---------------------------------------------------------------------------
# PR review — parallel_specialist as fallback
# ---------------------------------------------------------------------------


class TestPatternSelectorPRReview:
    def test_pr_review_agents_select_parallel_specialist(self, selector: PatternSelector):
        registry = CapabilityRegistry.build_pr_review_registry()
        agents = [
            _make_agent("pr_data_agent", ["fetch_pr_data", "fetch_github_diff", "fetch_changed_files"]),
            _make_agent("review_specialist", ["review_code_quality", "assess_architecture", "check_standards"]),
            _make_agent("risk_specialist", ["assess_security", "assess_testing", "assess_reliability"]),
            _make_agent("synthesis_agent", ["synthesize_findings", "produce_final_report"]),
        ]
        result = selector.select(_analysis(), agents, registry)
        assert result == "parallel_specialist"

    def test_non_triggering_agents_fall_back_to_parallel_specialist(self, selector: PatternSelector):
        registry = _full_registry()
        agents = [_make_agent("generic_agent", ["summarize", "fetch_data"])]
        result = selector.select(_analysis(), agents, registry)
        assert result == "parallel_specialist"

    def test_parallel_specialist_selected_when_no_trigger_caps_match(self, selector: PatternSelector):
        registry = _full_registry()
        agents = [_make_agent("agent_a", ["cap_x", "cap_y"])]
        result = selector.select(_analysis(), agents, registry)
        assert result == "parallel_specialist"


# ---------------------------------------------------------------------------
# Trigger-capability matching
# ---------------------------------------------------------------------------


class TestPatternSelectorTriggers:
    def test_agent_with_classify_intent_selects_router(self, selector: PatternSelector):
        registry = _full_registry()
        agents = [_make_agent("classifier", ["classify_intent", "route_request"])]
        result = selector.select(_analysis(), agents, registry)
        assert result == "router"

    def test_agent_with_iterative_research_selects_peo(self, selector: PatternSelector):
        registry = _full_registry()
        agents = [_make_agent("researcher", ["iterative_research", "explore"])]
        result = selector.select(_analysis(), agents, registry)
        assert result == "planner_executor_observer"

    def test_one_of_many_agents_has_trigger(self, selector: PatternSelector):
        registry = _full_registry()
        agents = [
            _make_agent("normal_agent", ["fetch_data"]),
            _make_agent("classifier", ["classify_intent"]),
        ]
        result = selector.select(_analysis(), agents, registry)
        assert result == "router"


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------


class TestPatternSelectorPriority:
    def test_router_takes_priority_over_peo_when_both_triggered(self, selector: PatternSelector):
        registry = _full_registry()
        agents = [_make_agent("dual_trigger", ["classify_intent", "iterative_research"])]
        result = selector.select(_analysis(), agents, registry)
        assert result == "router"

    def test_router_takes_priority_over_parallel_specialist(self, selector: PatternSelector):
        registry = _full_registry()
        agents = [
            _make_agent("classifier", ["classify_intent"]),
            _make_agent("generic", ["summarize"]),
        ]
        result = selector.select(_analysis(), agents, registry)
        assert result == "router"

    def test_peo_takes_priority_over_parallel_specialist(self, selector: PatternSelector):
        registry = _full_registry()
        agents = [_make_agent("researcher", ["iterative_research"])]
        result = selector.select(_analysis(), agents, registry)
        assert result == "planner_executor_observer"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestPatternSelectorEdgeCases:
    def test_pattern_not_in_registry_is_skipped(self, selector: PatternSelector):
        # Only parallel_specialist registered; router/peo priority entries are absent.
        registry = CapabilityRegistry()
        registry.register_pattern(PatternCapabilityDescriptor(
            pattern="parallel_specialist",
            name="Parallel Specialist",
            description="",
            best_for=[],
            supported_task_types=["code_review"],
            trigger_capabilities=[],
        ))
        # Even though the agent has classify_intent (which would trigger router),
        # router is not registered — so fallback to parallel_specialist.
        agents = [_make_agent("agent_a", ["classify_intent"])]
        result = selector.select(_analysis(), agents, registry)
        assert result == "parallel_specialist"

    def test_pattern_selection_is_capability_based(self, selector: PatternSelector):
        """PatternSelector uses agent.capabilities directly, not registry lookup."""
        registry = _full_registry()
        agents = [_make_agent("agent_a", ["some_cap"])]
        result = selector.select(_analysis(), agents, registry)
        assert result == "parallel_specialist"

    def test_agent_with_no_capabilities_contributes_nothing(self, selector: PatternSelector):
        """An agent with empty capabilities list contributes no trigger matches."""
        registry = _full_registry()
        agents = [_make_agent("empty_cap_agent", [])]
        result = selector.select(_analysis(), agents, registry)
        assert result == "parallel_specialist"

    def test_generated_agent_capabilities_are_used_for_pattern_selection(self, selector: PatternSelector):
        """Generated agents contribute their capabilities to trigger matching."""
        registry = _full_registry()
        generated = _make_agent("classify_agent", ["classify_intent"], generated=True)
        result = selector.select(_analysis(), [generated], registry)
        assert result == "router"

    def test_no_patterns_in_registry_returns_empty_string(self, selector: PatternSelector):
        registry = CapabilityRegistry()
        agents = [_make_agent("agent_a", ["some_cap"])]
        result = selector.select(_analysis(), agents, registry)
        assert result == ""

    def test_select_is_synchronous(self, selector: PatternSelector):
        import inspect
        assert not inspect.iscoroutinefunction(selector.select)

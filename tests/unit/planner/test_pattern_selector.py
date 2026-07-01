"""Unit tests for PatternSelector — deterministic, capability-driven."""

from __future__ import annotations

import pytest

from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.models import (
    AgentCapabilityDescriptor,
    GoalAnalysis,
    PatternCapabilityDescriptor,
    RiskLevel,
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


def _add_agent(registry: CapabilityRegistry, agent_id: str, capabilities: list[str]) -> None:
    registry.register_agent(AgentCapabilityDescriptor(
        agent_id=agent_id,
        name=agent_id,
        description="",
        capabilities=capabilities,
    ))


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
        agents = ["pr_data_agent", "review_specialist", "risk_specialist", "synthesis_agent"]
        result = selector.select(_analysis(), agents, registry)
        assert result == "parallel_specialist"

    def test_non_triggering_agents_fall_back_to_parallel_specialist(
        self, selector: PatternSelector
    ):
        registry = _full_registry()
        _add_agent(registry, "generic_agent", ["summarize", "fetch_data"])
        result = selector.select(_analysis(), ["generic_agent"], registry)
        assert result == "parallel_specialist"

    def test_parallel_specialist_selected_when_no_trigger_caps_match(
        self, selector: PatternSelector
    ):
        registry = _full_registry()
        _add_agent(registry, "agent_a", ["cap_x", "cap_y"])  # none of these trigger router/peo
        result = selector.select(_analysis(), ["agent_a"], registry)
        assert result == "parallel_specialist"


# ---------------------------------------------------------------------------
# Trigger-capability matching
# ---------------------------------------------------------------------------


class TestPatternSelectorTriggers:
    def test_agent_with_classify_intent_selects_router(self, selector: PatternSelector):
        registry = _full_registry()
        _add_agent(registry, "classifier", ["classify_intent", "route_request"])
        result = selector.select(_analysis(), ["classifier"], registry)
        assert result == "router"

    def test_agent_with_iterative_research_selects_peo(self, selector: PatternSelector):
        registry = _full_registry()
        _add_agent(registry, "researcher", ["iterative_research", "explore"])
        result = selector.select(_analysis(), ["researcher"], registry)
        assert result == "planner_executor_observer"

    def test_one_of_many_agents_has_trigger(self, selector: PatternSelector):
        registry = _full_registry()
        _add_agent(registry, "normal_agent", ["fetch_data"])
        _add_agent(registry, "classifier", ["classify_intent"])
        result = selector.select(_analysis(), ["normal_agent", "classifier"], registry)
        assert result == "router"


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------


class TestPatternSelectorPriority:
    def test_router_takes_priority_over_peo_when_both_triggered(self, selector: PatternSelector):
        registry = _full_registry()
        _add_agent(registry, "dual_trigger", ["classify_intent", "iterative_research"])
        result = selector.select(_analysis(), ["dual_trigger"], registry)
        assert result == "router"

    def test_router_takes_priority_over_parallel_specialist(self, selector: PatternSelector):
        registry = _full_registry()
        # Two agents: one triggers router, one would fall back to parallel_specialist
        _add_agent(registry, "classifier", ["classify_intent"])
        _add_agent(registry, "generic", ["summarize"])
        result = selector.select(_analysis(), ["classifier", "generic"], registry)
        assert result == "router"

    def test_peo_takes_priority_over_parallel_specialist(self, selector: PatternSelector):
        registry = _full_registry()
        _add_agent(registry, "researcher", ["iterative_research"])
        result = selector.select(_analysis(), ["researcher"], registry)
        assert result == "planner_executor_observer"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestPatternSelectorEdgeCases:
    def test_pattern_not_in_registry_is_skipped(self, selector: PatternSelector):
        # Only parallel_specialist registered; router/peo priority entries are absent
        registry = CapabilityRegistry()
        registry.register_pattern(PatternCapabilityDescriptor(
            pattern="parallel_specialist",
            name="Parallel Specialist",
            description="",
            best_for=[],
            supported_task_types=["code_review"],
            trigger_capabilities=[],
        ))
        _add_agent(registry, "agent_a", ["classify_intent"])  # trigger for router — but router not registered
        result = selector.select(_analysis(), ["agent_a"], registry)
        assert result == "parallel_specialist"

    def test_pattern_selection_is_capability_based(self, selector: PatternSelector):
        """PatternSelector uses agent capabilities, not goal classification, to pick a pattern."""
        registry = _full_registry()
        _add_agent(registry, "agent_a", ["some_cap"])
        # Analysis with no trigger-matching caps still falls back to parallel_specialist
        result = selector.select(_analysis(), ["agent_a"], registry)
        assert result == "parallel_specialist"

    def test_unknown_agent_id_contributes_no_caps(self, selector: PatternSelector):
        """Agent IDs not in registry are silently skipped when collecting caps."""
        registry = _full_registry()
        # "phantom_agent" is not registered — no caps contributed — only fallback matches
        result = selector.select(_analysis(), ["phantom_agent"], registry)
        assert result == "parallel_specialist"

    def test_no_patterns_in_registry_returns_empty_string(self, selector: PatternSelector):
        registry = CapabilityRegistry()
        _add_agent(registry, "agent_a", ["some_cap"])
        result = selector.select(_analysis(), ["agent_a"], registry)
        assert result == ""

    def test_select_is_synchronous(self, selector: PatternSelector):
        import inspect
        assert not inspect.iscoroutinefunction(selector.select)

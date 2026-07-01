"""Unit tests for GoalAnalyzer — deterministic via MockLLMProvider."""

from __future__ import annotations

import json

import pytest

from platform.core.interfaces.llm import ILLMProvider
from platform.core.models.message import LLMResponse, Message, StopReason, TextContent
from platform.core.models.tool import ToolDefinition
from platform.llm.mock_provider import MockLLMProvider
from platform.planner.capability_registry import CapabilityRegistry
from platform.planner.goal_analyzer import GoalAnalyzer
from platform.planner.models import GoalAnalysis, OperationType, PlannerError, RiskLevel, ToolCapabilityDescriptor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _response(text: str) -> LLMResponse:
    return LLMResponse(content=[TextContent(text=text)], stop_reason=StopReason.END_TURN)


def _json(**fields) -> str:
    base: dict = {
        "required_capabilities": ["fetch_pr_data", "review_code_quality"],
        "risk_level": "low",
        "confidence": 0.95,
        "reasoning": "Goal clearly maps to a GitHub PR review.",
        "constraints": ["read_only"],
        "requires_hitl": False,
    }
    base.update(fields)
    return json.dumps(base)


def _analyzer(response_text: str, registry: CapabilityRegistry | None = None) -> GoalAnalyzer:
    llm = MockLLMProvider([_response(response_text)])
    reg = registry or CapabilityRegistry.build_pr_review_registry()
    return GoalAnalyzer(llm=llm, registry=reg)


class _CapturingLLMProvider(ILLMProvider):
    """Records the messages passed to complete() for prompt inspection."""

    def __init__(self, response_text: str) -> None:
        self._response = _response(response_text)
        self.captured_messages: list[Message] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        self.captured_messages = list(messages)
        return self._response


# ---------------------------------------------------------------------------
# Happy path — PR review goal
# ---------------------------------------------------------------------------


class TestGoalAnalyzerPRReview:
    async def test_returns_goal_analysis_with_valid_capabilities(self):
        analyzer = _analyzer(_json())
        result = await analyzer.analyze("Review PR #42 for architecture and security")
        assert result.required_capabilities == ["fetch_pr_data", "review_code_quality"]

    async def test_returns_goal_analysis_instance(self):
        analyzer = _analyzer(_json())
        result = await analyzer.analyze("Review this GitHub PR")
        assert isinstance(result, GoalAnalysis)

    async def test_high_confidence_for_pr_review(self):
        analyzer = _analyzer(_json(confidence=0.95))
        result = await analyzer.analyze("Review PR #42")
        assert result.confidence == pytest.approx(0.95)

    async def test_low_risk_for_read_only_pr_review(self):
        analyzer = _analyzer(_json(risk_level="low"))
        result = await analyzer.analyze("Review PR #42")
        assert result.risk_level == RiskLevel.LOW

    async def test_does_not_require_hitl_for_low_risk(self):
        analyzer = _analyzer(_json(requires_hitl=False))
        result = await analyzer.analyze("Review PR #42")
        assert result.requires_hitl is False

    async def test_required_capabilities_extracted(self):
        caps = ["fetch_pr_data", "review_code_quality", "assess_security"]
        analyzer = _analyzer(_json(required_capabilities=caps))
        result = await analyzer.analyze("Review this PR for code quality and security")
        assert result.required_capabilities == caps

    async def test_constraints_extracted(self):
        analyzer = _analyzer(_json(constraints=["read_only", "no_external_writes"]))
        result = await analyzer.analyze("Review PR #42")
        assert "read_only" in result.constraints

    async def test_reasoning_is_a_string(self):
        analyzer = _analyzer(_json(reasoning="PR review goal is clear."))
        result = await analyzer.analyze("Review PR #42")
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0


# ---------------------------------------------------------------------------
# High-risk PR review — HITL enabled
# ---------------------------------------------------------------------------


class TestGoalAnalyzerHighRisk:
    async def test_high_risk_requires_hitl(self):
        analyzer = _analyzer(_json(risk_level="high", requires_hitl=True))
        result = await analyzer.analyze("Review PR #42 touching production payment service")
        assert result.risk_level == RiskLevel.HIGH
        assert result.requires_hitl is True

    async def test_critical_risk_requires_hitl(self):
        analyzer = _analyzer(_json(risk_level="critical", requires_hitl=True))
        result = await analyzer.analyze("Review PR #42 that deletes user accounts")
        assert result.risk_level == RiskLevel.CRITICAL
        assert result.requires_hitl is True

    async def test_medium_risk_hitl_false(self):
        analyzer = _analyzer(_json(risk_level="medium", requires_hitl=False))
        result = await analyzer.analyze("Review PR #42 in staging environment")
        assert result.risk_level == RiskLevel.MEDIUM
        assert result.requires_hitl is False


# ---------------------------------------------------------------------------
# Unsupported goal type
# ---------------------------------------------------------------------------


class TestGoalAnalyzerUnsupported:
    async def test_unsupported_goal_has_no_task_type_attribute(self):
        analyzer = _analyzer(
            _json(confidence=0, required_capabilities=[])
        )
        result = await analyzer.analyze("What is the capital of France?")
        assert not hasattr(result, "task_type")

    async def test_unsupported_goal_has_zero_confidence(self):
        analyzer = _analyzer(
            _json(task_type="unsupported", confidence=0.0, required_capabilities=[])
        )
        result = await analyzer.analyze("Send an email to the team")
        assert result.confidence == pytest.approx(0.0)

    async def test_unsupported_goal_has_empty_capabilities(self):
        analyzer = _analyzer(
            _json(task_type="unsupported", confidence=0, required_capabilities=[])
        )
        result = await analyzer.analyze("Search the web for latest news")
        assert result.required_capabilities == []


# ---------------------------------------------------------------------------
# Low confidence
# ---------------------------------------------------------------------------


class TestGoalAnalyzerLowConfidence:
    async def test_low_confidence_value_is_preserved(self):
        analyzer = _analyzer(_json(confidence=0.45))
        result = await analyzer.analyze("Do something with a PR maybe")
        assert result.confidence == pytest.approx(0.45)

    async def test_zero_point_one_confidence_is_valid(self):
        analyzer = _analyzer(_json(confidence=0.1, task_type="code_review"))
        result = await analyzer.analyze("Unclear goal involving a PR")
        assert result.confidence == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# JSON parsing errors → PlannerError
# ---------------------------------------------------------------------------


class TestGoalAnalyzerParsingErrors:
    async def test_malformed_json_raises_planner_error(self):
        analyzer = _analyzer("this is not JSON at all")
        with pytest.raises(PlannerError, match="invalid JSON"):
            await analyzer.analyze("Review PR #42")

    async def test_truncated_json_raises_planner_error(self):
        analyzer = _analyzer('{"task_type": "code_review"')
        with pytest.raises(PlannerError, match="invalid JSON"):
            await analyzer.analyze("Review PR #42")

    async def test_missing_required_capabilities_raises_planner_error(self):
        data = json.loads(_json())
        del data["required_capabilities"]
        analyzer = _analyzer(json.dumps(data))
        with pytest.raises(PlannerError, match="required_capabilities"):
            await analyzer.analyze("Review PR #42")

    async def test_missing_confidence_raises_planner_error(self):
        data = json.loads(_json())
        del data["confidence"]
        analyzer = _analyzer(json.dumps(data))
        with pytest.raises(PlannerError, match="confidence"):
            await analyzer.analyze("Review PR #42")

    async def test_missing_risk_level_raises_planner_error(self):
        data = json.loads(_json())
        del data["risk_level"]
        analyzer = _analyzer(json.dumps(data))
        with pytest.raises(PlannerError, match="risk_level"):
            await analyzer.analyze("Review PR #42")

    async def test_invalid_risk_level_enum_raises_planner_error(self):
        analyzer = _analyzer(_json(risk_level="extreme"))
        with pytest.raises(PlannerError, match="risk_level"):
            await analyzer.analyze("Review PR #42")

    async def test_confidence_above_1_raises_planner_error(self):
        analyzer = _analyzer(_json(confidence=1.5))
        with pytest.raises(PlannerError, match="confidence"):
            await analyzer.analyze("Review PR #42")

    async def test_confidence_below_0_raises_planner_error(self):
        analyzer = _analyzer(_json(confidence=-0.1))
        with pytest.raises(PlannerError, match="confidence"):
            await analyzer.analyze("Review PR #42")

    async def test_confidence_non_numeric_raises_planner_error(self):
        analyzer = _analyzer(_json(confidence="high"))
        with pytest.raises(PlannerError, match="confidence"):
            await analyzer.analyze("Review PR #42")

    async def test_empty_response_raises_planner_error(self):
        llm = MockLLMProvider([
            LLMResponse(content=[], stop_reason=StopReason.END_TURN)
        ])
        registry = CapabilityRegistry.build_pr_review_registry()
        analyzer = GoalAnalyzer(llm=llm, registry=registry)
        with pytest.raises(PlannerError, match="no text content"):
            await analyzer.analyze("Review PR #42")


# ---------------------------------------------------------------------------
# Markdown code fence stripping
# ---------------------------------------------------------------------------


class TestGoalAnalyzerCodeFenceStripping:
    async def test_strips_json_code_fence(self):
        fenced = f"```json\n{_json()}\n```"
        analyzer = _analyzer(fenced)
        result = await analyzer.analyze("Review PR #42")
        assert result.confidence == pytest.approx(0.95)

    async def test_strips_plain_code_fence(self):
        fenced = f"```\n{_json()}\n```"
        analyzer = _analyzer(fenced)
        result = await analyzer.analyze("Review PR #42")
        assert result.confidence == pytest.approx(0.95)

    async def test_strips_leading_whitespace(self):
        analyzer = _analyzer(f"\n\n{_json()}\n\n")
        result = await analyzer.analyze("Review PR #42")
        assert result.confidence == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Prompt content inspection
# ---------------------------------------------------------------------------


class TestGoalAnalyzerPrompt:
    async def test_prompt_contains_registry_summary(self):
        registry = CapabilityRegistry.build_pr_review_registry()
        capturing = _CapturingLLMProvider(_json())
        analyzer = GoalAnalyzer(llm=capturing, registry=registry)

        await analyzer.analyze("Review PR #42")

        system_message = capturing.captured_messages[0]
        assert system_message.role.value == "system"
        assert "pr_data_agent" in system_message.content
        assert "review_specialist" in system_message.content

    async def test_prompt_contains_supported_goal_types(self):
        registry = CapabilityRegistry.build_pr_review_registry()
        capturing = _CapturingLLMProvider(_json())
        analyzer = GoalAnalyzer(llm=capturing, registry=registry)

        await analyzer.analyze("Review PR #42")

        system_message = capturing.captured_messages[0]
        assert "code_review" in system_message.content

    async def test_prompt_does_not_classify_goals(self):
        """Phase C: prompt instructs LLM to identify capabilities, not classify goals."""
        registry = CapabilityRegistry.build_pr_review_registry()
        capturing = _CapturingLLMProvider(_json())
        analyzer = GoalAnalyzer(llm=capturing, registry=registry)

        await analyzer.analyze("Review PR #42")

        system_message = capturing.captured_messages[0]
        assert "Do not classify" in system_message.content

    async def test_user_message_contains_goal_text(self):
        registry = CapabilityRegistry.build_pr_review_registry()
        goal = "Review PR #99 for security vulnerabilities"
        capturing = _CapturingLLMProvider(_json())
        analyzer = GoalAnalyzer(llm=capturing, registry=registry)

        await analyzer.analyze(goal)

        user_message = capturing.captured_messages[1]
        assert user_message.role.value == "user"
        assert goal in user_message.content

    async def test_exactly_two_messages_sent(self):
        registry = CapabilityRegistry.build_pr_review_registry()
        capturing = _CapturingLLMProvider(_json())
        analyzer = GoalAnalyzer(llm=capturing, registry=registry)

        await analyzer.analyze("Review PR #42")

        assert len(capturing.captured_messages) == 2

    async def test_prompt_contains_tool_names_from_registry(self):
        registry = CapabilityRegistry.build_pr_review_registry()
        capturing = _CapturingLLMProvider(_json())
        analyzer = GoalAnalyzer(llm=capturing, registry=registry)

        await analyzer.analyze("Review PR #42")

        system_message = capturing.captured_messages[0]
        assert "github_get_pr" in system_message.content
        assert "knowledge_search" in system_message.content


# ---------------------------------------------------------------------------
# Edge cases and boundary values
# ---------------------------------------------------------------------------


class TestGoalAnalyzerEdgeCases:
    async def test_confidence_exactly_0_is_valid(self):
        analyzer = _analyzer(_json(task_type="unsupported", confidence=0))
        result = await analyzer.analyze("What is 2 + 2?")
        assert result.confidence == pytest.approx(0.0)

    async def test_confidence_exactly_1_is_valid(self):
        analyzer = _analyzer(_json(confidence=1.0))
        result = await analyzer.analyze("Review PR #42")
        assert result.confidence == pytest.approx(1.0)

    async def test_empty_constraints_list_is_valid(self):
        analyzer = _analyzer(_json(constraints=[]))
        result = await analyzer.analyze("Review PR #42")
        assert result.constraints == []

    async def test_integer_confidence_is_coerced_to_float(self):
        # JSON integers (0, 1) should be accepted and coerced
        analyzer = _analyzer(_json(task_type="unsupported", confidence=0))
        result = await analyzer.analyze("Something unrelated")
        assert isinstance(result.confidence, float)

    async def test_many_required_capabilities_preserved(self):
        caps = [
            "fetch_pr_data",
            "review_code_quality",
            "assess_security",
            "synthesize_findings",
            "assess_reliability",
        ]
        analyzer = _analyzer(_json(required_capabilities=caps))
        result = await analyzer.analyze("Full PR review")
        assert result.required_capabilities == caps


# ---------------------------------------------------------------------------
# Anti-hallucination filter (Phase C)
# ---------------------------------------------------------------------------


class TestGoalAnalyzerAntiHallucination:
    async def test_unknown_capability_moves_to_missing(self):
        caps = ["fetch_pr_data", "hallucinated_capability"]
        analyzer = _analyzer(_json(required_capabilities=caps))
        result = await analyzer.analyze("Review PR")
        assert "hallucinated_capability" not in result.required_capabilities
        assert "hallucinated_capability" in result.missing_capabilities
        assert "fetch_pr_data" in result.required_capabilities

    async def test_all_unknown_caps_move_to_missing(self):
        caps = ["filesystem_read", "summarization"]
        analyzer = _analyzer(_json(required_capabilities=caps))
        result = await analyzer.analyze("Summarize a file")
        assert result.required_capabilities == []
        assert set(result.missing_capabilities) == {"filesystem_read", "summarization"}

    async def test_known_caps_not_in_missing(self):
        caps = ["fetch_pr_data", "review_code_quality"]
        analyzer = _analyzer(_json(required_capabilities=caps))
        result = await analyzer.analyze("Review PR")
        assert result.missing_capabilities == []
        assert set(result.required_capabilities) == {"fetch_pr_data", "review_code_quality"}

    async def test_mixed_caps_split_correctly(self):
        caps = ["fetch_pr_data", "unknown_cap_x", "assess_security", "slack_notify"]
        analyzer = _analyzer(_json(required_capabilities=caps))
        result = await analyzer.analyze("Review PR with Slack notification")
        assert result.required_capabilities == ["fetch_pr_data", "assess_security"]
        assert result.missing_capabilities == ["unknown_cap_x", "slack_notify"]

    async def test_no_valid_caps_when_all_unknown(self):
        caps = ["filesystem_read", "summarization"]
        analyzer = _analyzer(_json(required_capabilities=caps))
        result = await analyzer.analyze("Summarize a file")
        assert result.required_capabilities == []

    async def test_some_valid_caps_when_mixed_known_unknown(self):
        caps = ["fetch_pr_data", "unknown_cap"]
        analyzer = _analyzer(_json(required_capabilities=caps))
        result = await analyzer.analyze("Review PR with unknown feature")
        assert len(result.required_capabilities) > 0

    async def test_missing_capabilities_empty_by_default(self):
        caps = ["fetch_pr_data", "review_code_quality"]
        analyzer = _analyzer(_json(required_capabilities=caps))
        result = await analyzer.analyze("Review PR")
        assert result.missing_capabilities == []

    async def test_missing_capabilities_field_exists_on_result(self):
        analyzer = _analyzer(_json())
        result = await analyzer.analyze("Review PR #42")
        assert hasattr(result, "missing_capabilities")
        assert isinstance(result.missing_capabilities, list)


# ---------------------------------------------------------------------------
# Capability deduplication (Phase C)
# ---------------------------------------------------------------------------


class TestGoalAnalyzerCapabilityDeduplication:
    async def test_duplicate_caps_deduplicated(self):
        caps = ["fetch_pr_data", "fetch_pr_data", "review_code_quality"]
        analyzer = _analyzer(_json(required_capabilities=caps))
        result = await analyzer.analyze("Review PR")
        assert result.required_capabilities == ["fetch_pr_data", "review_code_quality"]
        assert result.required_capabilities.count("fetch_pr_data") == 1

    async def test_dedup_preserves_order(self):
        caps = ["review_code_quality", "fetch_pr_data", "review_code_quality"]
        analyzer = _analyzer(_json(required_capabilities=caps))
        result = await analyzer.analyze("Review PR")
        assert result.required_capabilities == ["review_code_quality", "fetch_pr_data"]

    async def test_dedup_applies_before_filter(self):
        # Unknown duplicate should appear only once in missing_capabilities
        caps = ["unknown_cap", "unknown_cap", "fetch_pr_data"]
        analyzer = _analyzer(_json(required_capabilities=caps))
        result = await analyzer.analyze("PR plus unknown thing")
        assert result.missing_capabilities == ["unknown_cap"]
        assert result.required_capabilities == ["fetch_pr_data"]


# ---------------------------------------------------------------------------
# Capability allow-list injection (Phase C)
# ---------------------------------------------------------------------------


class TestGoalAnalyzerCapabilityAllowList:
    async def test_prompt_contains_all_agent_capabilities(self):
        registry = CapabilityRegistry.build_pr_review_registry()
        capturing = _CapturingLLMProvider(_json())
        analyzer = GoalAnalyzer(llm=capturing, registry=registry)

        await analyzer.analyze("Review PR #42")

        system_prompt = capturing.captured_messages[0].content
        for cap in registry.all_agent_capabilities():
            assert cap in system_prompt, f"Capability '{cap}' missing from system prompt"

    async def test_prompt_contains_allowlist_section(self):
        registry = CapabilityRegistry.build_pr_review_registry()
        capturing = _CapturingLLMProvider(_json())
        analyzer = GoalAnalyzer(llm=capturing, registry=registry)

        await analyzer.analyze("Review PR #42")

        system_prompt = capturing.captured_messages[0].content
        assert "allow-list" in system_prompt

    async def test_allowlist_contains_synthesis_and_risk_caps(self):
        registry = CapabilityRegistry.build_pr_review_registry()
        capturing = _CapturingLLMProvider(_json())
        analyzer = GoalAnalyzer(llm=capturing, registry=registry)

        await analyzer.analyze("Review PR #42")

        system_prompt = capturing.captured_messages[0].content
        assert "synthesize_findings" in system_prompt
        assert "assess_security" in system_prompt
        assert "fetch_pr_data" in system_prompt


# ---------------------------------------------------------------------------
# Unsupported goal scenarios (Phase C)
# ---------------------------------------------------------------------------


class TestGoalAnalyzerUnsupportedGoals:
    async def test_latency_monitoring_goal_empty_caps(self):
        analyzer = _analyzer(_json(required_capabilities=[], confidence=0.0))
        result = await analyzer.analyze("Monitor API latency over time")
        assert result.required_capabilities == []

    async def test_slack_notification_goal_empty_caps(self):
        analyzer = _analyzer(_json(required_capabilities=[], confidence=0.0))
        result = await analyzer.analyze("Send a Slack message to the team")
        assert result.required_capabilities == []

    async def test_email_goal_empty_caps(self):
        analyzer = _analyzer(_json(required_capabilities=[], confidence=0.0))
        result = await analyzer.analyze("Send weekly email digest")
        assert result.required_capabilities == []

    async def test_filesystem_summarization_unknown_caps_to_missing(self):
        caps = ["filesystem_read", "summarization"]
        analyzer = _analyzer(_json(required_capabilities=caps, confidence=0.5))
        result = await analyzer.analyze("Summarize all files in /docs")
        assert result.required_capabilities == []
        assert "filesystem_read" in result.missing_capabilities
        assert "summarization" in result.missing_capabilities

    async def test_knowledge_search_unknown_caps_to_missing(self):
        caps = ["knowledge_search", "summarization"]
        analyzer = _analyzer(_json(required_capabilities=caps, confidence=0.6))
        result = await analyzer.analyze("Search knowledge base and summarize findings")
        assert result.required_capabilities == []
        assert "knowledge_search" in result.missing_capabilities
        assert "summarization" in result.missing_capabilities


# ---------------------------------------------------------------------------
# Filesystem generatable capability (Phase D)
# ---------------------------------------------------------------------------


def _fs_registry() -> CapabilityRegistry:
    """PR review registry extended with filesystem_read as a generatable capability."""
    registry = CapabilityRegistry.build_pr_review_registry()
    registry.register_tool(ToolCapabilityDescriptor(
        tool_name="filesystem_read_file",
        name="Filesystem Read File",
        description="Reads a local file.",
        capabilities=["filesystem_read"],
        operation_type=OperationType.READ,
        data_source="local",
    ))
    registry.register_generatable_capability("filesystem_read")
    return registry


class TestGoalAnalyzerFilesystemCapability:
    async def test_filesystem_read_in_system_prompt_when_registered(self):
        capturing = _CapturingLLMProvider(
            _json(required_capabilities=["filesystem_read"])
        )
        analyzer = GoalAnalyzer(llm=capturing, registry=_fs_registry())
        await analyzer.analyze("Read README.md")
        system_prompt = capturing.captured_messages[0].content
        assert "filesystem_read" in system_prompt

    async def test_filesystem_read_passes_anti_hallucination_filter(self):
        llm = MockLLMProvider([_response(_json(required_capabilities=["filesystem_read"]))])
        result = await GoalAnalyzer(llm=llm, registry=_fs_registry()).analyze("Read README.md")
        assert "filesystem_read" in result.required_capabilities
        assert "filesystem_read" not in result.missing_capabilities

    async def test_filesystem_read_in_missing_without_registration(self):
        """Without explicit generatable registration, filesystem_read is filtered out."""
        result = await _analyzer(
            _json(required_capabilities=["filesystem_read"])
        ).analyze("Read README.md")
        assert "filesystem_read" not in result.required_capabilities
        assert "filesystem_read" in result.missing_capabilities

    async def test_all_agent_caps_still_present_with_filesystem_registered(self):
        registry = _fs_registry()
        capturing = _CapturingLLMProvider(_json())
        analyzer = GoalAnalyzer(llm=capturing, registry=registry)
        await analyzer.analyze("Review PR")
        system_prompt = capturing.captured_messages[0].content
        for cap in registry.all_agent_capabilities():
            assert cap in system_prompt

    async def test_filesystem_and_agent_cap_both_valid(self):
        llm = MockLLMProvider([_response(
            _json(required_capabilities=["fetch_pr_data", "filesystem_read"])
        )])
        result = await GoalAnalyzer(llm=llm, registry=_fs_registry()).analyze(
            "Review PR and read README"
        )
        assert "fetch_pr_data" in result.required_capabilities
        assert "filesystem_read" in result.required_capabilities
        assert result.missing_capabilities == []

"""Unit tests for ResultAggregator."""

from __future__ import annotations

import pytest

from platform.aggregator.result_aggregator import AggregationStrategy, ResultAggregator
from platform.core.models.agent import AgentResult


def make_result(agent_id: str, output: str) -> AgentResult:
    return AgentResult(agent_id=agent_id, output=output)


class TestResultAggregator:
    def test_concatenate_multiple_results(self):
        aggregator = ResultAggregator()
        results = [
            make_result("security-agent", "No critical vulnerabilities found."),
            make_result("performance-agent", "Response time within SLA."),
        ]
        output = aggregator.aggregate(results, AggregationStrategy.CONCATENATE)
        assert "## [security-agent]" in output
        assert "No critical vulnerabilities found." in output
        assert "## [performance-agent]" in output
        assert "Response time within SLA." in output

    def test_concatenate_single_result(self):
        aggregator = ResultAggregator()
        results = [make_result("agent-1", "Single output.")]
        output = aggregator.aggregate(results, AggregationStrategy.CONCATENATE)
        assert "## [agent-1]" in output
        assert "Single output." in output

    def test_concatenate_empty_list_returns_empty_string(self):
        aggregator = ResultAggregator()
        output = aggregator.aggregate([], AggregationStrategy.CONCATENATE)
        assert output == ""

    def test_default_strategy_is_concatenate(self):
        aggregator = ResultAggregator()
        results = [make_result("agent-1", "hello")]
        output = aggregator.aggregate(results)
        assert "## [agent-1]" in output

    def test_concatenate_sections_separated_by_blank_line(self):
        aggregator = ResultAggregator()
        results = [
            make_result("a1", "output one"),
            make_result("a2", "output two"),
        ]
        output = aggregator.aggregate(results, AggregationStrategy.CONCATENATE)
        assert "\n\n" in output

    def test_concatenate_preserves_agent_order(self):
        aggregator = ResultAggregator()
        results = [
            make_result("first", "1st"),
            make_result("second", "2nd"),
            make_result("third", "3rd"),
        ]
        output = aggregator.aggregate(results, AggregationStrategy.CONCATENATE)
        pos_first = output.index("[first]")
        pos_second = output.index("[second]")
        pos_third = output.index("[third]")
        assert pos_first < pos_second < pos_third

    def test_synthesize_raises_not_implemented(self):
        aggregator = ResultAggregator()
        results = [make_result("agent-1", "output")]
        with pytest.raises(NotImplementedError):
            aggregator.aggregate(results, AggregationStrategy.SYNTHESIZE)

    def test_synthesize_raises_even_with_llm_provider_none(self):
        aggregator = ResultAggregator(llm_provider=None)
        with pytest.raises(NotImplementedError):
            aggregator.aggregate(
                [make_result("a", "b")], AggregationStrategy.SYNTHESIZE
            )

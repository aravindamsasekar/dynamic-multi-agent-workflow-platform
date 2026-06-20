"""ResultAggregator — merges outputs from parallel specialist agents."""

from __future__ import annotations

from enum import Enum

from platform.core.models.agent import AgentResult


class AggregationStrategy(str, Enum):
    CONCATENATE = "concatenate"
    SYNTHESIZE = "synthesize"


class ResultAggregator:
    """Combines a list of AgentResults into a single output string.

    Strategies:
        CONCATENATE — Join results with labeled agent headers. No LLM call required.
        SYNTHESIZE  — Call LLM to write a unified summary (stub; requires llm_provider).
    """

    def __init__(self, llm_provider: object | None = None) -> None:
        self._llm_provider = llm_provider

    def aggregate(
        self,
        results: list[AgentResult],
        strategy: AggregationStrategy = AggregationStrategy.CONCATENATE,
    ) -> str:
        """Merge agent results into a single output string using the given strategy."""
        if strategy == AggregationStrategy.CONCATENATE:
            return self._concatenate(results)
        if strategy == AggregationStrategy.SYNTHESIZE:
            raise NotImplementedError(
                "SYNTHESIZE requires an LLM provider and will be implemented in Phase 3"
            )
        raise ValueError(f"Unknown aggregation strategy: {strategy}")

    def _concatenate(self, results: list[AgentResult]) -> str:
        """Join results as labeled markdown sections."""
        if not results:
            return ""
        sections = [f"## [{r.agent_id}]\n\n{r.output}" for r in results]
        return "\n\n".join(sections)

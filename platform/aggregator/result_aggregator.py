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
        SYNTHESIZE  — Call LLM to write a unified summary from all results.
    """

    def __init__(self, llm_provider: object | None = None) -> None:
        self._llm_provider = llm_provider

    def aggregate(
        self,
        results: list[AgentResult],
        strategy: AggregationStrategy = AggregationStrategy.CONCATENATE,
    ) -> str:
        """Merge agent results into a single output string using the given strategy."""
        # TODO: implement
        raise NotImplementedError

"""Deterministic task label inference from a set of selected capabilities."""

from __future__ import annotations


class TaskLabelInferer:
    """Infers a human-readable task label from a capability list.

    Rules are checked in order; the first rule whose required capability set
    is a subset of the input set wins. Returns "custom" if no rule matches.

    No LLM. No registry lookup. Pure deterministic matching.
    """

    _RULES: tuple[tuple[frozenset[str], str], ...] = (
        # PR review (current V3 registry capability names)
        (frozenset({"fetch_pr_data", "review_code_quality"}), "code_review"),
        # Document comparison — checked before file_analysis (more specific)
        (frozenset({"filesystem_read", "knowledge_search", "comparison"}), "document_comparison"),
        # File analysis
        (frozenset({"filesystem_read", "summarization"}), "file_analysis"),
        # Knowledge query
        (frozenset({"knowledge_search", "summarization"}), "knowledge_query"),
        # Alternative PR review capability names (future capabilities)
        (frozenset({"github_pr_read", "review_code", "security_review"}), "code_review"),
    )

    def infer(self, capabilities: list[str]) -> str:
        """Return a task label for the given capability list.

        Subset matching: all caps in a rule must be present in the input.
        Returns "custom" when no rule matches.
        """
        cap_set = frozenset(capabilities)
        for required, label in self._RULES:
            if required.issubset(cap_set):
                return label
        return "custom"

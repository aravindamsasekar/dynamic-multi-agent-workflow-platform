"""Unit tests for TaskLabelInferer — deterministic, no LLM."""

from __future__ import annotations

import inspect

from platform.planner.task_label_inferer import TaskLabelInferer


# ---------------------------------------------------------------------------
# Known rule mappings
# ---------------------------------------------------------------------------


class TestTaskLabelInfererKnownMappings:
    def test_pr_review_capabilities_map_to_code_review(self):
        inferer = TaskLabelInferer()
        result = inferer.infer(["fetch_pr_data", "review_code_quality", "assess_security", "synthesize_findings"])
        assert result == "code_review"

    def test_pr_review_minimal_capabilities_map_to_code_review(self):
        inferer = TaskLabelInferer()
        result = inferer.infer(["fetch_pr_data", "review_code_quality"])
        assert result == "code_review"

    def test_pr_review_superset_still_matches_code_review(self):
        inferer = TaskLabelInferer()
        result = inferer.infer(["fetch_pr_data", "review_code_quality", "extra_cap", "another"])
        assert result == "code_review"

    def test_file_analysis_capabilities(self):
        inferer = TaskLabelInferer()
        result = inferer.infer(["filesystem_read", "summarization"])
        assert result == "file_analysis"

    def test_knowledge_query_capabilities(self):
        inferer = TaskLabelInferer()
        result = inferer.infer(["knowledge_search", "summarization"])
        assert result == "knowledge_query"

    def test_document_comparison_capabilities(self):
        inferer = TaskLabelInferer()
        result = inferer.infer(["filesystem_read", "knowledge_search", "comparison"])
        assert result == "document_comparison"

    def test_document_comparison_superset_matches(self):
        inferer = TaskLabelInferer()
        result = inferer.infer(["filesystem_read", "knowledge_search", "comparison", "extra"])
        assert result == "document_comparison"

    def test_unknown_capabilities_return_custom(self):
        inferer = TaskLabelInferer()
        result = inferer.infer(["unknown_cap_a", "unknown_cap_b"])
        assert result == "custom"

    def test_empty_capabilities_return_custom(self):
        inferer = TaskLabelInferer()
        result = inferer.infer([])
        assert result == "custom"

    def test_single_known_cap_without_matching_rule_returns_custom(self):
        inferer = TaskLabelInferer()
        # "fetch_pr_data" alone doesn't satisfy any complete rule
        result = inferer.infer(["fetch_pr_data"])
        assert result == "custom"

    def test_partial_rule_match_returns_custom(self):
        inferer = TaskLabelInferer()
        # filesystem_read alone doesn't satisfy file_analysis (needs summarization too)
        result = inferer.infer(["filesystem_read"])
        assert result == "custom"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestTaskLabelInfererDeterminism:
    def test_same_input_same_output_code_review(self):
        inferer = TaskLabelInferer()
        caps = ["fetch_pr_data", "review_code_quality"]
        assert inferer.infer(caps) == inferer.infer(caps)

    def test_same_input_same_output_custom(self):
        inferer = TaskLabelInferer()
        caps = ["made_up_cap"]
        assert inferer.infer(caps) == inferer.infer(caps)

    def test_two_instances_produce_same_result(self):
        caps = ["fetch_pr_data", "review_code_quality"]
        assert TaskLabelInferer().infer(caps) == TaskLabelInferer().infer(caps)

    def test_input_order_does_not_affect_result(self):
        inferer = TaskLabelInferer()
        caps_a = ["filesystem_read", "summarization"]
        caps_b = ["summarization", "filesystem_read"]
        assert inferer.infer(caps_a) == inferer.infer(caps_b)

    def test_infer_is_synchronous(self):
        assert not inspect.iscoroutinefunction(TaskLabelInferer().infer)


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------


class TestTaskLabelInfererPriority:
    def test_document_comparison_takes_priority_over_file_analysis(self):
        inferer = TaskLabelInferer()
        # Has both filesystem_read+summarization (file_analysis) AND
        # filesystem_read+knowledge_search+comparison (document_comparison).
        # document_comparison rule is checked first — it wins.
        result = inferer.infer(["filesystem_read", "knowledge_search", "comparison", "summarization"])
        assert result == "document_comparison"

    def test_document_comparison_takes_priority_over_knowledge_query(self):
        inferer = TaskLabelInferer()
        # Has both knowledge_search+summarization (knowledge_query) AND
        # filesystem_read+knowledge_search+comparison (document_comparison).
        result = inferer.infer(["filesystem_read", "knowledge_search", "comparison", "summarization"])
        assert result == "document_comparison"

    def test_pr_review_rule_checked_before_fallback(self):
        inferer = TaskLabelInferer()
        result = inferer.infer(["fetch_pr_data", "review_code_quality"])
        assert result != "custom"

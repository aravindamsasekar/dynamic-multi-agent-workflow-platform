"""Unit tests for PolicyEngine and ContentFilterRule."""

from __future__ import annotations

import pytest

from platform.core.exceptions import PolicyViolation
from platform.core.interfaces.policy import IRule, PolicyDecision
from platform.policy.engine import HookPoint, PolicyEngine
from platform.policy.rules.content_filter import ContentFilterRule


# ---------------------------------------------------------------------------
# ContentFilterRule
# ---------------------------------------------------------------------------


class TestContentFilterRule:
    def test_blocks_when_term_found(self):
        rule = ContentFilterRule(blocked_terms=["badword"])
        ctx = {"content": "this contains badword inside"}
        assert rule.check(ctx) == PolicyDecision.BLOCK

    def test_allows_clean_content(self):
        rule = ContentFilterRule(blocked_terms=["badword"])
        ctx = {"content": "this is perfectly fine"}
        assert rule.check(ctx) == PolicyDecision.ALLOW

    def test_case_insensitive_matching(self):
        rule = ContentFilterRule(blocked_terms=["badword"])
        assert rule.check({"content": "BADWORD"}) == PolicyDecision.BLOCK
        assert rule.check({"content": "BadWord"}) == PolicyDecision.BLOCK
        assert rule.check({"content": "bAdWoRd"}) == PolicyDecision.BLOCK

    def test_empty_blocked_terms_always_allows(self):
        rule = ContentFilterRule(blocked_terms=[])
        assert rule.check({"content": "anything goes here"}) == PolicyDecision.ALLOW

    def test_no_blocked_terms_arg_always_allows(self):
        rule = ContentFilterRule()
        assert rule.check({"content": "anything"}) == PolicyDecision.ALLOW

    def test_blocks_on_first_matching_term(self):
        rule = ContentFilterRule(blocked_terms=["term1", "term2"])
        assert rule.check({"content": "contains term1"}) == PolicyDecision.BLOCK
        assert rule.check({"content": "contains term2"}) == PolicyDecision.BLOCK

    def test_missing_content_key_does_not_raise(self):
        rule = ContentFilterRule(blocked_terms=["bad"])
        assert rule.check({}) == PolicyDecision.ALLOW

    def test_none_content_value_does_not_raise(self):
        rule = ContentFilterRule(blocked_terms=["bad"])
        assert rule.check({"content": None}) == PolicyDecision.ALLOW

    def test_blocked_term_stored_lowercased(self):
        rule = ContentFilterRule(blocked_terms=["UPPER"])
        assert rule.check({"content": "upper text"}) == PolicyDecision.BLOCK


# ---------------------------------------------------------------------------
# PolicyEngine
# ---------------------------------------------------------------------------


class _AllowRule(IRule):
    def check(self, context: dict) -> PolicyDecision:
        return PolicyDecision.ALLOW


class _BlockRule(IRule):
    def check(self, context: dict) -> PolicyDecision:
        return PolicyDecision.BLOCK


class _WarnRule(IRule):
    def check(self, context: dict) -> PolicyDecision:
        return PolicyDecision.WARN


class TestPolicyEngine:
    def test_evaluate_raises_on_block(self):
        engine = PolicyEngine(rules=[_BlockRule()])
        with pytest.raises(PolicyViolation):
            engine.evaluate(HookPoint.PRE_AGENT, {"content": "test"})

    def test_evaluate_passes_when_all_allow(self):
        engine = PolicyEngine(rules=[_AllowRule(), _AllowRule()])
        engine.evaluate(HookPoint.PRE_AGENT, {"content": "clean"})  # must not raise

    def test_evaluate_passes_with_no_rules(self):
        engine = PolicyEngine()
        engine.evaluate(HookPoint.PRE_TOOL, {"content": "anything"})  # must not raise

    def test_evaluate_warn_does_not_raise(self):
        engine = PolicyEngine(rules=[_WarnRule()])
        engine.evaluate(HookPoint.POST_AGENT, {"content": "warn me"})  # must not raise

    def test_evaluate_blocks_even_if_some_rules_allow(self):
        engine = PolicyEngine(rules=[_AllowRule(), _BlockRule(), _AllowRule()])
        with pytest.raises(PolicyViolation):
            engine.evaluate(HookPoint.PRE_AGENT, {"content": "test"})

    def test_add_rule_registers_rule(self):
        engine = PolicyEngine()
        engine.add_rule(_BlockRule())
        with pytest.raises(PolicyViolation):
            engine.evaluate(HookPoint.PRE_AGENT, {"content": "test"})

    def test_violation_message_includes_hook_name(self):
        engine = PolicyEngine(rules=[_BlockRule()])
        with pytest.raises(PolicyViolation, match="pre_agent"):
            engine.evaluate(HookPoint.PRE_AGENT, {"content": "test"})

    def test_content_filter_integration(self):
        engine = PolicyEngine(rules=[ContentFilterRule(blocked_terms=["forbidden"])])
        engine.evaluate(HookPoint.PRE_AGENT, {"content": "safe text"})  # must not raise
        with pytest.raises(PolicyViolation):
            engine.evaluate(HookPoint.PRE_AGENT, {"content": "this is forbidden"})

"""Unit tests for PolicyEngine and ContentFilterRule."""

import pytest


class TestContentFilterRule:
    # TODO: test check() returns BLOCK when blocked term is found in content
    # TODO: test check() returns ALLOW for clean content
    # TODO: test check() is case-insensitive
    # TODO: test check() with empty blocked_terms list always returns ALLOW


class TestPolicyEngine:
    # TODO: test evaluate() raises PolicyViolation when any rule returns BLOCK
    # TODO: test evaluate() passes silently when all rules return ALLOW
    # TODO: test evaluate() with no registered rules always passes
    # TODO: test add_rule() registers additional rules
    pass

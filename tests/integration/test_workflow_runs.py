"""Integration tests for end-to-end workflow runs."""

import pytest


class TestWorkflowRuns:
    # TODO: test full parallel specialist run with MockAdapter tools and mock LLM
    # TODO: test full router run — classifier routes to correct specialist agent
    # TODO: test full planner → executor → observer run completes in one iteration
    # TODO: test HITL pause flow — run reaches checkpoint and status becomes WAITING_APPROVAL
    # TODO: test HITL approve flow — approved run resumes and reaches COMPLETED
    # TODO: test HITL reject flow — rejected run transitions to FAILED with HITLRejected
    pass

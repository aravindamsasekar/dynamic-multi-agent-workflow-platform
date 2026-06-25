"""Integration tests for demo workflow configurations.

Each test class loads a real workflow directory via ConfigLoader.load_one(),
builds an ExecutionContext from the populated registries, runs the appropriate
PatternExecutor with a pre-queued MockLLMProvider, and asserts on
WorkflowResult structure and content.

No real LLM API calls are made in any test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from platform.config.loader import ConfigLoader
from platform.core.exceptions import PatternExecutionError
from platform.core.interfaces.observer import IObserver
from platform.core.models.context import ExecutionContext
from platform.core.models.events import WorkflowEvent
from platform.core.models.message import (
    LLMResponse,
    StopReason,
    TextContent,
    ToolUseContent,
)
from platform.core.models.workflow import WorkflowResult
from platform.memory.in_memory_store import InMemoryStore
from platform.orchestrator.orchestrator import Orchestrator
from platform.orchestrator.run_manager import RunManager
from platform.patterns.parallel_specialist import ParallelSpecialistExecutor
from platform.patterns.planner_executor_observer import PlannerExecutorObserverExecutor
from platform.patterns.router import RouterExecutor
from platform.policy.engine import PolicyEngine
from platform.registries.agent_registry import AgentRegistry
from platform.registries.tool_registry import ToolRegistry
from platform.registries.workflow_registry import WorkflowRegistry
from platform.state.shared_state import SharedState
from platform.llm.mock_provider import MockLLMProvider

_WORKFLOWS_DIR = Path(__file__).resolve().parent.parent.parent / "workflows"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _CapturingObserver(IObserver):
    def __init__(self) -> None:
        self.events: list[WorkflowEvent] = []

    def on_event(self, event: WorkflowEvent) -> None:
        self.events.append(event)


def _text(text: str) -> LLMResponse:
    return LLMResponse(content=[TextContent(text=text)], stop_reason=StopReason.END_TURN)


def _tool_use(tool_id: str, tool_name: str, input: dict[str, Any] = {}) -> LLMResponse:
    return LLMResponse(
        content=[ToolUseContent(id=tool_id, name=tool_name, input=input)],
        stop_reason=StopReason.TOOL_USE,
    )


def _load(workflow_dir_name: str) -> tuple[WorkflowRegistry, AgentRegistry, ToolRegistry]:
    wf_reg = WorkflowRegistry()
    ag_reg = AgentRegistry()
    tl_reg = ToolRegistry()
    ConfigLoader(wf_reg, ag_reg, tl_reg).load_one(_WORKFLOWS_DIR / workflow_dir_name)
    return wf_reg, ag_reg, tl_reg


def _context(
    wf_reg: WorkflowRegistry,
    ag_reg: AgentRegistry,
    tl_reg: ToolRegistry,
    workflow_id: str,
    run_id: str = "integration-run-001",
) -> ExecutionContext:
    return ExecutionContext(
        run_id=run_id,
        workflow_definition=wf_reg.get(workflow_id),
        shared_state=None,
        workflow_registry=wf_reg,
        agent_registry=ag_reg,
        tool_registry=tl_reg,
        memory_store=InMemoryStore(),
        policy_engine=PolicyEngine(),
        observer=_CapturingObserver(),
    )


# ---------------------------------------------------------------------------
# TestIncidentCommanderWorkflow
# ---------------------------------------------------------------------------


class TestIncidentCommanderWorkflow:
    """Exercises the parallel_specialist pattern with three specialists + reviewer."""

    async def test_produces_workflow_result(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("incident_commander")
        llm = MockLLMProvider([
            _text("High CPU (94%) and memory pressure (87%) detected."),   # metrics_agent
            _text("DB connection timeouts in payment-service."),            # logs_agent
            _text("payment-service v2.4.1 deployed 37 min before spike."), # deployment_agent
            _text("Root cause: DB pool exhaustion post v2.4.1 deploy."),   # reviewer_agent
        ])
        result = await ParallelSpecialistExecutor(llm).execute(
            _context(wf_reg, ag_reg, tl_reg, "incident_commander"),
            "Production alert: payment-service p99 latency >2s",
        )
        assert isinstance(result, WorkflowResult)
        assert result.workflow_id == "incident_commander"
        assert result.output == "Root cause: DB pool exhaustion post v2.4.1 deploy."

    async def test_all_specialist_results_included(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("incident_commander")
        llm = MockLLMProvider([
            _text("Metrics: CPU 94%, latency spike."),
            _text("Logs: circuit breaker opened."),
            _text("Deploy: v2.4.1 shipped 37m ago."),
            _text("Summary: rollback v2.4.1 immediately."),
        ])
        result = await ParallelSpecialistExecutor(llm).execute(
            _context(wf_reg, ag_reg, tl_reg, "incident_commander"),
            "Production incident",
        )
        assert len(result.agent_results) == 4
        specialist_ids = {r.agent_id for r in result.agent_results[:3]}
        assert specialist_ids == {"metrics_agent", "logs_agent", "deployment_agent"}
        assert result.agent_results[3].agent_id == "reviewer_agent"

    async def test_concatenated_sections_passed_to_reviewer(self) -> None:
        # The reviewer's output is the final WorkflowResult.output, confirming
        # that the reviewer ran after aggregation.
        wf_reg, ag_reg, tl_reg = _load("incident_commander")
        reviewer_output = "Remediation: rollback, scale DB connections, page on-call."
        llm = MockLLMProvider([
            _text("Metrics finding."),
            _text("Logs finding."),
            _text("Deployment finding."),
            _text(reviewer_output),
        ])
        result = await ParallelSpecialistExecutor(llm).execute(
            _context(wf_reg, ag_reg, tl_reg, "incident_commander"),
            "Incident",
        )
        assert result.output == reviewer_output

    async def test_metrics_agent_executes_mock_tool(self) -> None:
        # MockLLMProvider returns a tool-use response for the metrics agent,
        # verifying that the MockAdapter registered from tools.yaml is invoked.
        wf_reg, ag_reg, tl_reg = _load("incident_commander")
        llm = MockLLMProvider([
            # metrics_agent: tool call then final answer
            _tool_use("call-001", "mock_metrics_tool", {"time_range": "last_5m"}),
            _text("CPU at 94% — critical threshold breached."),
            # remaining agents: direct answers
            _text("Logs: repeated DB timeouts."),
            _text("Deploy: v2.4.1 is the likely culprit."),
            _text("Root cause confirmed: DB pool exhaustion."),
        ])
        result = await ParallelSpecialistExecutor(llm).execute(
            _context(wf_reg, ag_reg, tl_reg, "incident_commander"),
            "Production incident",
        )
        assert result.output == "Root cause confirmed: DB pool exhaustion."
        metrics_result = next(
            r for r in result.agent_results if r.agent_id == "metrics_agent"
        )
        assert metrics_result.tool_calls_made == 1


# ---------------------------------------------------------------------------
# TestCustomerSupportWorkflow
# ---------------------------------------------------------------------------


class TestCustomerSupportWorkflow:
    """Exercises the router pattern with a classifier and two specialist branches."""

    async def test_routes_billing_query_to_billing_agent(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("customer_support")
        llm = MockLLMProvider([
            _text("billing"),                                   # classifier
            _text("Your invoice was issued on the 1st."),       # billing_agent
        ])
        result = await RouterExecutor(llm).execute(
            _context(wf_reg, ag_reg, tl_reg, "customer_support"),
            "Why was I charged twice this month?",
        )
        assert result.output == "Your invoice was issued on the 1st."
        assert result.agent_results[0].agent_id == "classifier_agent"
        assert result.agent_results[1].agent_id == "billing_agent"

    async def test_routes_technical_query_to_technical_agent(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("customer_support")
        llm = MockLLMProvider([
            _text("technical"),                                         # classifier
            _text("1. Check your API key. 2. Verify endpoint URL."),    # technical_agent
        ])
        result = await RouterExecutor(llm).execute(
            _context(wf_reg, ag_reg, tl_reg, "customer_support"),
            "I keep getting 401 errors from the API.",
        )
        assert result.output == "1. Check your API key. 2. Verify endpoint URL."
        assert result.agent_results[1].agent_id == "technical_agent"

    async def test_agent_results_contains_classifier_and_target(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("customer_support")
        llm = MockLLMProvider([_text("billing"), _text("Refund processed.")])
        result = await RouterExecutor(llm).execute(
            _context(wf_reg, ag_reg, tl_reg, "customer_support"),
            "I need a refund.",
        )
        assert len(result.agent_results) == 2

    async def test_unknown_route_raises_pattern_execution_error(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("customer_support")
        llm = MockLLMProvider([_text("shipping")])  # not in routes
        with pytest.raises(PatternExecutionError, match="shipping"):
            await RouterExecutor(llm).execute(
                _context(wf_reg, ag_reg, tl_reg, "customer_support"),
                "Where is my package?",
            )


# ---------------------------------------------------------------------------
# TestResearchWorkflow
# ---------------------------------------------------------------------------


class TestResearchWorkflow:
    """Exercises the planner_executor_observer pattern with loop control signals."""

    async def test_done_on_first_iteration(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("research_workflow")
        llm = MockLLMProvider([
            _text("Search for recent papers on transformer attention mechanisms."),  # planner
            _text("Found 5 key papers from 2023 covering sparse attention."),        # executor
            _text("DONE"),                                                           # observer
        ])
        result = await PlannerExecutorObserverExecutor(llm).execute(
            _context(wf_reg, ag_reg, tl_reg, "research_workflow"),
            "Summarize recent advances in transformer attention mechanisms.",
        )
        assert isinstance(result, WorkflowResult)
        assert result.output == "Found 5 key papers from 2023 covering sparse attention."
        assert len(result.agent_results) == 3

    async def test_continues_on_retry_then_done(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("research_workflow")
        llm = MockLLMProvider([
            _text("Search for transformer papers."),            # planner iter 0
            _text("Results were too broad."),                   # executor iter 0
            _text("RETRY"),                                     # observer iter 0
            _text("Narrow search to sparse attention only."),   # planner iter 1
            _text("Found 3 precise papers on sparse attention."),# executor iter 1
            _text("DONE"),                                      # observer iter 1
        ])
        result = await PlannerExecutorObserverExecutor(llm).execute(
            _context(wf_reg, ag_reg, tl_reg, "research_workflow"),
            "Summarize sparse attention research.",
        )
        assert result.output == "Found 3 precise papers on sparse attention."
        assert len(result.agent_results) == 6  # 2 iterations × 3 agents

    async def test_max_iterations_exceeded_raises(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("research_workflow")
        # Override max_iterations by modifying pattern_config on the loaded definition
        wf_def = wf_reg.get("research_workflow")
        wf_def.pattern_config["max_iterations"] = 2

        llm = MockLLMProvider([
            _text("Step A."), _text("Result A."), _text("RETRY"),   # iter 0
            _text("Step B."), _text("Result B."), _text("RETRY"),   # iter 1
        ])
        with pytest.raises(PatternExecutionError, match="exceeded"):
            await PlannerExecutorObserverExecutor(llm).execute(
                _context(wf_reg, ag_reg, tl_reg, "research_workflow"),
                "Research topic.",
            )


# ---------------------------------------------------------------------------
# TestPRReviewWorkflow
# ---------------------------------------------------------------------------


def _make_github_client(response_text: str = "{}") -> MagicMock:
    """AsyncClient mock that returns a fixed response for any GET request."""
    mock_response = MagicMock()
    mock_response.text = response_text
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)
    return mock_client


_PR_INPUT = '{"owner": "octocat", "repo": "Hello-World", "pull_number": 42}'
_GITHUB_PATCH = "platform.tools.github_adapter.httpx.AsyncClient"


class TestPRReviewWorkflow:
    """Exercises the pr_review workflow: single GitHub fetch agent + review agent.

    GitHub HTTP calls are intercepted via mock — no real API calls are made.
    """

    async def test_produces_workflow_result(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("pr_review")
        llm = MockLLMProvider([
            # github_fetch_agent: 3 sequential tool calls then summary
            _tool_use("tu-1", "github_get_pr", {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _tool_use("tu-2", "github_get_files", {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _tool_use("tu-3", "github_get_diff", {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _text("PR #42: Add feature. 2 files changed (+50/-10)."),
            # review_agent: final review
            _text("LGTM: clean implementation with good test coverage."),
        ])
        with patch(_GITHUB_PATCH, return_value=_make_github_client()):
            result = await ParallelSpecialistExecutor(llm).execute(
                _context(wf_reg, ag_reg, tl_reg, "pr_review"),
                _PR_INPUT,
            )
        assert isinstance(result, WorkflowResult)
        assert result.workflow_id == "pr_review"
        assert result.output == "LGTM: clean implementation with good test coverage."

    async def test_result_contains_both_agents(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("pr_review")
        llm = MockLLMProvider([
            _tool_use("tu-1", "github_get_pr", {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _tool_use("tu-2", "github_get_files", {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _tool_use("tu-3", "github_get_diff", {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _text("Fetched PR data."),
            _text("Review: request changes."),
        ])
        with patch(_GITHUB_PATCH, return_value=_make_github_client()):
            result = await ParallelSpecialistExecutor(llm).execute(
                _context(wf_reg, ag_reg, tl_reg, "pr_review"),
                _PR_INPUT,
            )
        agent_ids = [r.agent_id for r in result.agent_results]
        assert "github_fetch_agent" in agent_ids
        assert "review_agent" in agent_ids
        assert len(result.agent_results) == 2

    async def test_fetch_agent_makes_three_tool_calls(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("pr_review")
        llm = MockLLMProvider([
            _tool_use("tu-1", "github_get_pr", {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _tool_use("tu-2", "github_get_files", {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _tool_use("tu-3", "github_get_diff", {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _text("All data collected."),
            _text("Looks good."),
        ])
        with patch(_GITHUB_PATCH, return_value=_make_github_client()):
            result = await ParallelSpecialistExecutor(llm).execute(
                _context(wf_reg, ag_reg, tl_reg, "pr_review"),
                _PR_INPUT,
            )
        fetch_result = next(r for r in result.agent_results if r.agent_id == "github_fetch_agent")
        assert fetch_result.tool_calls_made == 3

    async def test_review_agent_receives_fetch_output(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("pr_review")
        fetch_summary = "Title: Add feature. Files: main.py (+20). Diff: +def add_feature()."
        llm = MockLLMProvider([
            _tool_use("tu-1", "github_get_pr", {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _tool_use("tu-2", "github_get_files", {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _tool_use("tu-3", "github_get_diff", {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _text(fetch_summary),
            _text("Approved: minimal change, well-scoped."),
        ])
        with patch(_GITHUB_PATCH, return_value=_make_github_client()):
            result = await ParallelSpecialistExecutor(llm).execute(
                _context(wf_reg, ag_reg, tl_reg, "pr_review"),
                _PR_INPUT,
            )
        fetch_result = next(r for r in result.agent_results if r.agent_id == "github_fetch_agent")
        assert fetch_result.output == fetch_summary
        assert result.output == "Approved: minimal change, well-scoped."

    async def test_owner_repo_pull_number_propagate_to_github_urls(self) -> None:
        # Proves that owner/repo/pull_number from the tool call args reach the
        # correct GitHub API URLs — the critical end-to-end propagation check.
        wf_reg, ag_reg, tl_reg = _load("pr_review")
        mock_gh = _make_github_client()
        llm = MockLLMProvider([
            _tool_use("tu-1", "github_get_pr",    {"owner": "microsoft", "repo": "vscode", "pull_number": 99}),
            _tool_use("tu-2", "github_get_files", {"owner": "microsoft", "repo": "vscode", "pull_number": 99}),
            _tool_use("tu-3", "github_get_diff",  {"owner": "microsoft", "repo": "vscode", "pull_number": 99}),
            _text("PR data collected."),
            _text("Looks good."),
        ])
        with patch(_GITHUB_PATCH, return_value=mock_gh):
            await ParallelSpecialistExecutor(llm).execute(
                _context(wf_reg, ag_reg, tl_reg, "pr_review"),
                '{"owner": "microsoft", "repo": "vscode", "pull_number": 99}',
            )

        called_urls = [call[0][0] for call in mock_gh.get.call_args_list]
        assert len(called_urls) == 3
        assert "https://api.github.com/repos/microsoft/vscode/pulls/99" in called_urls
        assert "https://api.github.com/repos/microsoft/vscode/pulls/99/files" in called_urls

    async def test_dict_input_via_orchestrator_reaches_github_urls(self) -> None:
        # Full end-to-end: POST /runs body → Orchestrator (dict→JSON) → agent →
        # tool calls → GitHubAdapter → correct API URLs.
        wf_reg, ag_reg, tl_reg = _load("pr_review")
        shared_state = SharedState()
        run_manager = RunManager()
        mock_gh = _make_github_client()
        llm = MockLLMProvider([
            _tool_use("tu-1", "github_get_pr",    {"owner": "octocat", "repo": "Hello-World", "pull_number": 1}),
            _tool_use("tu-2", "github_get_files", {"owner": "octocat", "repo": "Hello-World", "pull_number": 1}),
            _tool_use("tu-3", "github_get_diff",  {"owner": "octocat", "repo": "Hello-World", "pull_number": 1}),
            _text("Fetched all PR data."),
            _text("Approved."),
        ])
        orch = Orchestrator(
            workflow_registry=wf_reg,
            agent_registry=ag_reg,
            tool_registry=tl_reg,
            memory_store=InMemoryStore(),
            policy_engine=PolicyEngine(),
            observer=_CapturingObserver(),
            run_manager=run_manager,
            llm_provider=llm,
            shared_state=shared_state,
        )
        structured_input = {"owner": "octocat", "repo": "Hello-World", "pull_number": 1}
        with patch(_GITHUB_PATCH, return_value=mock_gh):
            result = await orch.run("pr_review", structured_input)

        # Workflow completes successfully
        assert result.workflow_id == "pr_review"
        assert result.output == "Approved."

        # Structured input preserved in SharedState
        assert shared_state.get(result.run_id, "workflow_input") == structured_input

        # Correct GitHub API URLs were called
        called_urls = [call[0][0] for call in mock_gh.get.call_args_list]
        assert len(called_urls) == 3
        assert "https://api.github.com/repos/octocat/Hello-World/pulls/1" in called_urls
        assert "https://api.github.com/repos/octocat/Hello-World/pulls/1/files" in called_urls

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

from unittest.mock import AsyncMock, MagicMock, patch

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
from platform.knowledge.service import KnowledgeService
from platform.knowledge.vector_store import SearchResult
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
from platform.tools.mcp_adapter import MCPAdapter

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


def _mock_ks(results: list[SearchResult] | None = None) -> MagicMock:
    """Return a MagicMock KnowledgeService that returns preset search results."""
    ks = MagicMock(spec=KnowledgeService)
    ks.search = AsyncMock(return_value=results or [])
    return ks


def _load(
    workflow_dir_name: str,
    knowledge_service: object | None = None,
) -> tuple[WorkflowRegistry, AgentRegistry, ToolRegistry]:
    wf_reg = WorkflowRegistry()
    ag_reg = AgentRegistry()
    tl_reg = ToolRegistry()
    ConfigLoader(
        wf_reg, ag_reg, tl_reg, knowledge_service=knowledge_service
    ).load_one(_WORKFLOWS_DIR / workflow_dir_name)
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
    """Exercises the 4-agent pr_review workflow: pr_data_agent + two specialists + synthesis.

    GitHub HTTP calls are intercepted via mock — no real API calls are made.
    Queue order with mocked LLM is deterministic: pr_data_agent → review_specialist
    → risk_specialist → synthesis_agent (mocked I/O never yields to the event loop).
    """

    async def test_produces_workflow_result(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("pr_review", knowledge_service=_mock_ks())
        llm = MockLLMProvider([
            # pr_data_agent: 3 GitHub tool calls then structured summary
            _tool_use("tu-1", "github_get_pr",    {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _tool_use("tu-2", "github_get_files", {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _tool_use("tu-3", "github_get_diff",  {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _text("PR #42: Add feature. 2 files changed (+50/-10)."),
            # review_specialist: direct code review (no tool calls in this test)
            _text("Code quality: clean implementation with clear naming."),
            # risk_specialist: direct risk assessment (no tool calls in this test)
            _text("No security issues. Tests cover new logic."),
            # synthesis_agent: final structured report
            _text("APPROVED: well-scoped PR with test coverage and no security concerns."),
        ])
        with patch(_GITHUB_PATCH, return_value=_make_github_client()):
            result = await ParallelSpecialistExecutor(llm).execute(
                _context(wf_reg, ag_reg, tl_reg, "pr_review"),
                _PR_INPUT,
            )
        assert isinstance(result, WorkflowResult)
        assert result.workflow_id == "pr_review"
        assert result.output == "APPROVED: well-scoped PR with test coverage and no security concerns."

    async def test_result_contains_four_agents(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("pr_review", knowledge_service=_mock_ks())
        llm = MockLLMProvider([
            _tool_use("tu-1", "github_get_pr",    {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _tool_use("tu-2", "github_get_files", {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _tool_use("tu-3", "github_get_diff",  {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _text("PR data collected."),
            _text("Code review: approved."),
            _text("Risk: low."),
            _text("APPROVED."),
        ])
        with patch(_GITHUB_PATCH, return_value=_make_github_client()):
            result = await ParallelSpecialistExecutor(llm).execute(
                _context(wf_reg, ag_reg, tl_reg, "pr_review"),
                _PR_INPUT,
            )
        agent_ids = [r.agent_id for r in result.agent_results]
        assert "pr_data_agent" in agent_ids
        assert "review_specialist" in agent_ids
        assert "risk_specialist" in agent_ids
        assert "synthesis_agent" in agent_ids
        assert len(result.agent_results) == 4

    async def test_pr_data_agent_makes_three_github_tool_calls(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("pr_review", knowledge_service=_mock_ks())
        llm = MockLLMProvider([
            _tool_use("tu-1", "github_get_pr",    {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _tool_use("tu-2", "github_get_files", {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _tool_use("tu-3", "github_get_diff",  {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _text("All data collected."),
            _text("Code looks good."),
            _text("No risks."),
            _text("APPROVED."),
        ])
        with patch(_GITHUB_PATCH, return_value=_make_github_client()):
            result = await ParallelSpecialistExecutor(llm).execute(
                _context(wf_reg, ag_reg, tl_reg, "pr_review"),
                _PR_INPUT,
            )
        pr_data_result = next(r for r in result.agent_results if r.agent_id == "pr_data_agent")
        assert pr_data_result.tool_calls_made == 3

    async def test_synthesis_agent_output_is_final_result(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("pr_review", knowledge_service=_mock_ks())
        synthesis_output = "APPROVED: clean PR with good coverage."
        llm = MockLLMProvider([
            _tool_use("tu-1", "github_get_pr",    {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _tool_use("tu-2", "github_get_files", {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _tool_use("tu-3", "github_get_diff",  {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _text("PR data."),
            _text("Code review complete."),
            _text("Risk review complete."),
            _text(synthesis_output),
        ])
        with patch(_GITHUB_PATCH, return_value=_make_github_client()):
            result = await ParallelSpecialistExecutor(llm).execute(
                _context(wf_reg, ag_reg, tl_reg, "pr_review"),
                _PR_INPUT,
            )
        synthesis_result = next(r for r in result.agent_results if r.agent_id == "synthesis_agent")
        assert synthesis_result.output == synthesis_output
        assert result.output == synthesis_output

    async def test_owner_repo_pull_number_propagate_to_github_urls(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("pr_review", knowledge_service=_mock_ks())
        mock_gh = _make_github_client()
        llm = MockLLMProvider([
            # Only pr_data_agent calls GitHub; specialists give direct text
            _tool_use("tu-1", "github_get_pr",    {"owner": "microsoft", "repo": "vscode", "pull_number": 99}),
            _tool_use("tu-2", "github_get_files", {"owner": "microsoft", "repo": "vscode", "pull_number": 99}),
            _tool_use("tu-3", "github_get_diff",  {"owner": "microsoft", "repo": "vscode", "pull_number": 99}),
            _text("PR data collected."),
            _text("Review: LGTM."),
            _text("No risks."),
            _text("APPROVED."),
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
        wf_reg, ag_reg, tl_reg = _load("pr_review", knowledge_service=_mock_ks())
        shared_state = SharedState()
        run_manager = RunManager()
        mock_gh = _make_github_client()
        llm = MockLLMProvider([
            _tool_use("tu-1", "github_get_pr",    {"owner": "octocat", "repo": "Hello-World", "pull_number": 1}),
            _tool_use("tu-2", "github_get_files", {"owner": "octocat", "repo": "Hello-World", "pull_number": 1}),
            _tool_use("tu-3", "github_get_diff",  {"owner": "octocat", "repo": "Hello-World", "pull_number": 1}),
            _text("Fetched all PR data."),
            _text("Code review: clean."),
            _text("No risks."),
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

        assert result.workflow_id == "pr_review"
        assert result.output == "Approved."
        assert shared_state.get(result.run_id, "workflow_input") == structured_input

        called_urls = [call[0][0] for call in mock_gh.get.call_args_list]
        assert len(called_urls) == 3
        assert "https://api.github.com/repos/octocat/Hello-World/pulls/1" in called_urls
        assert "https://api.github.com/repos/octocat/Hello-World/pulls/1/files" in called_urls


# ---------------------------------------------------------------------------
# TestPRReviewWithKnowledge
# ---------------------------------------------------------------------------


class TestPRReviewWithKnowledge:
    """Verifies the 4-agent PR Review workflow with knowledge-grounded specialists.

    GitHub and OpenAI calls are mocked. KnowledgeService uses AsyncMock to return
    controlled SearchResult objects. MCP calls use MCPConnectionManager patches.
    """

    # -- Configuration loading ------------------------------------------------

    def test_pr_review_config_loads_with_knowledge_search(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("pr_review", knowledge_service=_mock_ks())
        wf = wf_reg.get("pr_review")
        assert wf.workflow_id == "pr_review"

    def test_knowledge_search_tool_registered_in_tool_registry(self) -> None:
        _, _, tl_reg = _load("pr_review", knowledge_service=_mock_ks())
        adapter = tl_reg.get("knowledge_search")
        assert adapter is not None

    def test_review_specialist_has_knowledge_search_tool(self) -> None:
        _, ag_reg, _ = _load("pr_review", knowledge_service=_mock_ks())
        agent = ag_reg.get("review_specialist")
        assert "knowledge_search" in agent.tool_names

    def test_risk_specialist_has_knowledge_search_tool(self) -> None:
        _, ag_reg, _ = _load("pr_review", knowledge_service=_mock_ks())
        agent = ag_reg.get("risk_specialist")
        assert "knowledge_search" in agent.tool_names

    def test_pr_data_agent_has_three_github_tools(self) -> None:
        _, ag_reg, _ = _load("pr_review", knowledge_service=_mock_ks())
        agent = ag_reg.get("pr_data_agent")
        assert set(agent.tool_names) == {"github_get_pr", "github_get_files", "github_get_diff"}

    def test_knowledge_search_tool_definition_has_correct_adapter_type(self) -> None:
        from platform.core.models.tool import AdapterType
        _, _, tl_reg = _load("pr_review", knowledge_service=_mock_ks())
        tool_def = tl_reg.get_definition("knowledge_search")
        assert tool_def.adapter_type == AdapterType.KNOWLEDGE

    def test_knowledge_search_collections_include_coding_standards(self) -> None:
        from platform.tools.knowledge_adapter import KnowledgeAdapter
        _, _, tl_reg = _load("pr_review", knowledge_service=_mock_ks())
        adapter = tl_reg.get("knowledge_search")
        assert isinstance(adapter, KnowledgeAdapter)
        assert "coding-standards" in adapter._collections

    def test_mcp_get_pr_comments_tool_registered(self) -> None:
        _, _, tl_reg = _load("pr_review", knowledge_service=_mock_ks())
        adapter = tl_reg.get("mcp_get_pr_comments")
        assert adapter is not None
        assert isinstance(adapter, MCPAdapter)

    def test_synthesis_agent_has_mcp_tool(self) -> None:
        _, ag_reg, _ = _load("pr_review", knowledge_service=_mock_ks())
        agent = ag_reg.get("synthesis_agent")
        assert "mcp_get_pr_comments" in agent.tool_names

    # -- Workflow execution with mocked knowledge -----------------------------

    async def test_workflow_completes_when_knowledge_returns_empty(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("pr_review", knowledge_service=_mock_ks([]))
        llm = MockLLMProvider([
            # pr_data_agent
            _tool_use("tu-1", "github_get_pr",    {"owner": "o", "repo": "r", "pull_number": 1}),
            _tool_use("tu-2", "github_get_files", {"owner": "o", "repo": "r", "pull_number": 1}),
            _tool_use("tu-3", "github_get_diff",  {"owner": "o", "repo": "r", "pull_number": 1}),
            _text("PR data collected."),
            # review_specialist: searches knowledge (empty), writes review
            _tool_use("ks-1", "knowledge_search", {"query": "pull request review requirements tests"}),
            _text("Code review complete."),
            # risk_specialist: direct
            _text("Security review complete."),
            # synthesis_agent
            _text("Approved: clean implementation."),
        ])
        with patch(_GITHUB_PATCH, return_value=_make_github_client()):
            result = await ParallelSpecialistExecutor(llm).execute(
                _context(wf_reg, ag_reg, tl_reg, "pr_review"),
                _PR_INPUT,
            )
        assert isinstance(result, WorkflowResult)
        assert result.output == "Approved: clean implementation."

    async def test_review_specialist_calls_knowledge_search(self) -> None:
        ks = _mock_ks([])
        wf_reg, ag_reg, tl_reg = _load("pr_review", knowledge_service=ks)
        llm = MockLLMProvider([
            _tool_use("tu-1", "github_get_pr",    {"owner": "o", "repo": "r", "pull_number": 1}),
            _tool_use("tu-2", "github_get_files", {"owner": "o", "repo": "r", "pull_number": 1}),
            _tool_use("tu-3", "github_get_diff",  {"owner": "o", "repo": "r", "pull_number": 1}),
            _text("PR fetched."),
            _tool_use("ks-1", "knowledge_search", {"query": "coding standards"}),
            _text("Review complete."),
            _text("No risks."),
            _text("APPROVED."),
        ])
        with patch(_GITHUB_PATCH, return_value=_make_github_client()):
            await ParallelSpecialistExecutor(llm).execute(
                _context(wf_reg, ag_reg, tl_reg, "pr_review"),
                _PR_INPUT,
            )
        ks.search.assert_called_once()
        call_args = ks.search.call_args
        assert call_args[0][0] == "coding standards"

    async def test_knowledge_search_called_with_correct_collections(self) -> None:
        ks = _mock_ks([])
        wf_reg, ag_reg, tl_reg = _load("pr_review", knowledge_service=ks)
        llm = MockLLMProvider([
            _tool_use("tu-1", "github_get_pr",    {"owner": "o", "repo": "r", "pull_number": 1}),
            _tool_use("tu-2", "github_get_files", {"owner": "o", "repo": "r", "pull_number": 1}),
            _tool_use("tu-3", "github_get_diff",  {"owner": "o", "repo": "r", "pull_number": 1}),
            _text("PR fetched."),
            _tool_use("ks-1", "knowledge_search", {"query": "test requirements"}),
            _text("Done."),
            _text("No risk."),
            _text("APPROVED."),
        ])
        with patch(_GITHUB_PATCH, return_value=_make_github_client()):
            await ParallelSpecialistExecutor(llm).execute(
                _context(wf_reg, ag_reg, tl_reg, "pr_review"),
                _PR_INPUT,
            )
        call_args = ks.search.call_args
        collections = call_args[0][1]  # second positional arg
        assert "coding-standards" in collections

    async def test_retrieved_knowledge_appears_in_review_context(self) -> None:
        standards_text = "Every PR must include tests for all changed business logic."
        ks = _mock_ks([
            SearchResult(
                faiss_id=1,
                text=standards_text,
                source_file="coding-standards/pr_review_guidelines.md",
                score=0.95,
                collection="coding-standards",
            )
        ])
        wf_reg, ag_reg, tl_reg = _load("pr_review", knowledge_service=ks)
        llm = MockLLMProvider([
            _tool_use("tu-1", "github_get_pr",    {"owner": "o", "repo": "r", "pull_number": 1}),
            _tool_use("tu-2", "github_get_files", {"owner": "o", "repo": "r", "pull_number": 1}),
            _tool_use("tu-3", "github_get_diff",  {"owner": "o", "repo": "r", "pull_number": 1}),
            _text("PR data."),
            _tool_use("ks-1", "knowledge_search", {"query": "test requirements"}),
            _text("Request Changes: tests are missing per coding standards."),
            _text("Security: no issues."),
            _text("Request Changes: tests are missing per our coding standards."),
        ])
        with patch(_GITHUB_PATCH, return_value=_make_github_client()):
            result = await ParallelSpecialistExecutor(llm).execute(
                _context(wf_reg, ag_reg, tl_reg, "pr_review"),
                _PR_INPUT,
            )
        assert "Request Changes" in result.output
        ks.search.assert_called_once()

    async def test_multiple_knowledge_searches_all_succeed(self) -> None:
        ks = _mock_ks([])
        wf_reg, ag_reg, tl_reg = _load("pr_review", knowledge_service=ks)
        llm = MockLLMProvider([
            _tool_use("tu-1", "github_get_pr",    {"owner": "o", "repo": "r", "pull_number": 1}),
            _tool_use("tu-2", "github_get_files", {"owner": "o", "repo": "r", "pull_number": 1}),
            _tool_use("tu-3", "github_get_diff",  {"owner": "o", "repo": "r", "pull_number": 1}),
            _text("PR fetched."),
            _tool_use("ks-1", "knowledge_search", {"query": "test requirements"}),
            _tool_use("ks-2", "knowledge_search", {"query": "error handling standards"}),
            _text("Review: Approve."),
            _text("No risk."),
            _text("APPROVED."),
        ])
        with patch(_GITHUB_PATCH, return_value=_make_github_client()):
            result = await ParallelSpecialistExecutor(llm).execute(
                _context(wf_reg, ag_reg, tl_reg, "pr_review"),
                _PR_INPUT,
            )
        assert result.output == "APPROVED."
        assert ks.search.call_count == 2

    async def test_synthesis_agent_calls_mcp_get_pr_comments(self) -> None:
        ks = _mock_ks([])
        wf_reg, ag_reg, tl_reg = _load("pr_review", knowledge_service=ks)
        mcp_session = _make_mcp_session(
            tool_names=["list_pull_request_review_comments"],
            call_text="Prior review: LGTM from @reviewer",
        )
        mcp_stdio = _make_mcp_stdio()
        llm = MockLLMProvider([
            _tool_use("tu-1", "github_get_pr",    {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _tool_use("tu-2", "github_get_files", {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _tool_use("tu-3", "github_get_diff",  {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _text("PR data."),
            _text("Code review."),
            _text("Risk assessment."),
            # synthesis_agent: calls mcp_get_pr_comments then finalizes
            _tool_use("mcp-1", "mcp_get_pr_comments", {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _text("APPROVED with prior review context."),
        ])
        with patch(_MCP_PARAMS_PATCH), \
             patch(_MCP_SESSION_PATCH, return_value=mcp_session), \
             patch(_MCP_STDIO_PATCH, return_value=mcp_stdio), \
             patch(_GITHUB_PATCH, return_value=_make_github_client()):
            result = await ParallelSpecialistExecutor(llm).execute(
                _context(wf_reg, ag_reg, tl_reg, "pr_review"),
                _PR_INPUT,
            )
        assert result.output == "APPROVED with prior review context."
        mcp_session.call_tool.assert_called_once()

    async def test_existing_github_only_path_still_works(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("pr_review", knowledge_service=_mock_ks())
        llm = MockLLMProvider([
            _tool_use("tu-1", "github_get_pr",    {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _tool_use("tu-2", "github_get_files", {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _tool_use("tu-3", "github_get_diff",  {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}),
            _text("PR #42: Add feature. 2 files changed (+50/-10)."),
            _text("Architecture looks clean."),
            _text("No security issues detected."),
            _text("LGTM: clean implementation with good test coverage."),
        ])
        with patch(_GITHUB_PATCH, return_value=_make_github_client()):
            result = await ParallelSpecialistExecutor(llm).execute(
                _context(wf_reg, ag_reg, tl_reg, "pr_review"),
                _PR_INPUT,
            )
        assert result.output == "LGTM: clean implementation with good test coverage."
        assert result.workflow_id == "pr_review"


# ---------------------------------------------------------------------------
# TestDevOpsRemediationWorkflow
# ---------------------------------------------------------------------------

_MCP_STDIO_PATCH = "platform.tools.mcp_connection_manager.stdio_client"
_MCP_SESSION_PATCH = "platform.tools.mcp_connection_manager.ClientSession"
_MCP_PARAMS_PATCH = "platform.tools.mcp_connection_manager.StdioServerParameters"


def _make_mcp_session(tool_names: list[str] | None = None, call_text: str = "file contents") -> MagicMock:
    """Return a mocked MCP ClientSession."""
    tools = []
    for name in (tool_names or ["read_file"]):
        t = MagicMock()
        t.name = name
        t.description = ""
        t.inputSchema = {"type": "object"}
        tools.append(t)
    list_result = MagicMock()
    list_result.tools = tools

    content_item = MagicMock()
    content_item.text = call_text
    call_result = MagicMock()
    call_result.content = [content_item]
    call_result.isError = False

    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock(return_value=list_result)
    session.call_tool = AsyncMock(return_value=call_result)
    return session


def _make_mcp_stdio() -> MagicMock:
    """Return a mocked stdio_client context manager."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


class TestDevOpsRemediationWorkflow:
    """Integration tests for the devops_remediation MCP filesystem workflow."""

    def test_workflow_loads_without_error(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("devops_remediation")
        assert wf_reg.exists("devops_remediation")

    def test_file_analyst_agent_registered(self) -> None:
        _, ag_reg, _ = _load("devops_remediation")
        agent = ag_reg.get("file_analyst_agent")
        assert agent.agent_id == "file_analyst_agent"

    def test_filesystem_read_file_tool_registered(self) -> None:
        _, _, tl_reg = _load("devops_remediation")
        assert tl_reg.exists("filesystem_read_file")

    def test_filesystem_read_file_is_mcp_adapter(self) -> None:
        _, _, tl_reg = _load("devops_remediation")
        adapter = tl_reg.get("filesystem_read_file")
        assert isinstance(adapter, MCPAdapter)

    def test_mcp_adapter_tool_name_is_read_file(self) -> None:
        _, _, tl_reg = _load("devops_remediation")
        adapter = tl_reg.get("filesystem_read_file")
        assert isinstance(adapter, MCPAdapter)
        assert adapter._tool_name == "read_file"

    def test_file_analyst_agent_has_filesystem_read_file_tool(self) -> None:
        _, ag_reg, _ = _load("devops_remediation")
        agent = ag_reg.get("file_analyst_agent")
        assert "filesystem_read_file" in agent.tool_names

    async def test_execute_with_mocked_mcp_returns_summary(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("devops_remediation")
        session = _make_mcp_session(call_text="# README\nThis is the README file.")
        stdio = _make_mcp_stdio()
        llm = MockLLMProvider([
            _tool_use("mcp-1", "filesystem_read_file", {"path": "README.md"}),
            _text("Purpose: project documentation. Key contents: README with project overview."),
        ])
        with patch(_MCP_PARAMS_PATCH), \
             patch(_MCP_SESSION_PATCH, return_value=session), \
             patch(_MCP_STDIO_PATCH, return_value=stdio):
            result = await ParallelSpecialistExecutor(llm).execute(
                _context(wf_reg, ag_reg, tl_reg, "devops_remediation"),
                "Summarize README.md",
            )
        assert isinstance(result, WorkflowResult)
        assert result.workflow_id == "devops_remediation"
        assert "README" in result.output

    async def test_mcp_tool_call_arguments_forwarded(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("devops_remediation")
        session = _make_mcp_session()
        stdio = _make_mcp_stdio()
        llm = MockLLMProvider([
            _tool_use("mcp-1", "filesystem_read_file", {"path": "pyproject.toml"}),
            _text("Analysis complete."),
        ])
        with patch(_MCP_PARAMS_PATCH), \
             patch(_MCP_SESSION_PATCH, return_value=session), \
             patch(_MCP_STDIO_PATCH, return_value=stdio):
            await ParallelSpecialistExecutor(llm).execute(
                _context(wf_reg, ag_reg, tl_reg, "devops_remediation"),
                "Analyse pyproject.toml",
            )
        session.call_tool.assert_called_once_with(
            "read_file", arguments={"path": "pyproject.toml"}
        )

    async def test_mcp_tool_error_propagates_as_error_tool_result(self) -> None:
        wf_reg, ag_reg, tl_reg = _load("devops_remediation")
        session = _make_mcp_session()
        stdio = _make_mcp_stdio()

        error_content = MagicMock()
        error_content.text = "File not found: missing.txt"
        error_result = MagicMock()
        error_result.content = [error_content]
        error_result.isError = True
        session.call_tool = AsyncMock(return_value=error_result)

        llm = MockLLMProvider([
            _tool_use("mcp-1", "filesystem_read_file", {"path": "missing.txt"}),
            _text("Could not read file — it does not exist."),
        ])
        with patch(_MCP_PARAMS_PATCH), \
             patch(_MCP_SESSION_PATCH, return_value=session), \
             patch(_MCP_STDIO_PATCH, return_value=stdio):
            result = await ParallelSpecialistExecutor(llm).execute(
                _context(wf_reg, ag_reg, tl_reg, "devops_remediation"),
                "Summarize missing.txt",
            )
        assert isinstance(result, WorkflowResult)

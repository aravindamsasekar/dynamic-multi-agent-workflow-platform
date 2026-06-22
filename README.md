# Dynamic Multi-Agent Workflow Platform

A Python platform for building and running multi-agent AI workflows. Define workflows in YAML, run them via a REST API, and extend with new patterns or LLM providers without changing core platform code.

## Architecture

```
api/                        # FastAPI layer (endpoints, schemas, DI)
platform/
  agent/                    # AgentRuntime — drives one agent through LLM + tool loop
  aggregator/               # ResultAggregator — concatenate agent outputs
  config/                   # ConfigLoader, ConfigValidator — parse workflow YAML
  core/
    exceptions.py           # Platform exceptions
    interfaces/             # ILLMProvider, IToolAdapter, IObserver, IPolicyEngine, …
    models/                 # WorkflowDefinition, AgentDefinition, ExecutionContext, …
  hitl/                     # ApprovalManager — pause/resume for human-in-the-loop
  llm/                      # OpenAIProvider, MockLLMProvider
  memory/                   # InMemoryStore (IMemoryStore)
  observability/            # ConsoleObserver — emits JSON event lines to stdout
  orchestrator/             # Orchestrator, RunManager
  patterns/                 # ParallelSpecialistExecutor, RouterExecutor, PEOExecutor
  policy/                   # PolicyEngine + ContentFilterRule
  registries/               # WorkflowRegistry, AgentRegistry, ToolRegistry
  state/                    # SharedState — per-run cross-agent key-value store
  tools/                    # MockAdapter, HTTPAdapter, MCPAdapter
workflows/                  # YAML workflow definitions (one folder per workflow)
tests/
  unit/                     # Fast unit tests — no real LLM calls
  integration/              # End-to-end pattern tests using real YAML + MockLLMProvider
```

## Supported Patterns

| Pattern | YAML `pattern` value | Description |
|---|---|---|
| Parallel Specialist | `parallel_specialist` | Run N agents concurrently, aggregate outputs, optionally pass to a reviewer |
| Router | `router` | Classifier agent picks a route label, target agent handles the request |
| Planner-Executor-Observer | `planner_executor_observer` | Iterative loop: planner → executor → observer until DONE signal or max iterations |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and set your API key:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini        # optional, defaults to gpt-4o-mini
```

## Running Tests

```bash
pytest
```

All tests use `MockLLMProvider` — no real API key required.

## Starting the API Server

```bash
uvicorn api.main:app --reload
```

The server requires `OPENAI_API_KEY` to be set. It will refuse to start without it.

Interactive docs: http://localhost:8000/docs

## Demo Workflows

### List all workflows

```bash
curl http://localhost:8000/workflows/
```

```json
[
  {"workflow_id": "incident_commander", "name": "Incident Commander", ...},
  {"workflow_id": "customer_support",   "name": "Customer Support Router", ...},
  {"workflow_id": "research_workflow",  "name": "Research Workflow", ...}
]
```

### Incident Commander — parallel_specialist

Three specialist agents (metrics, logs, deployment) run in parallel. A reviewer synthesizes their findings.

```bash
curl -s -X POST http://localhost:8000/runs/ \
  -H "Content-Type: application/json" \
  -d '{"workflow_id": "incident_commander", "input": "Production alert: payment-service p99 latency spiked to 4s"}' \
  | python -m json.tool
```

```json
{
  "run_id": "b3f2...",
  "workflow_id": "incident_commander",
  "status": "completed",
  "output": "Root cause: DB connection pool exhausted after v2.4.1 deploy. Recommended action: rollback and scale pool size."
}
```

### Customer Support Router — router

A classifier routes the query to either the billing or technical specialist.

```bash
curl -s -X POST http://localhost:8000/runs/ \
  -H "Content-Type: application/json" \
  -d '{"workflow_id": "customer_support", "input": "I was charged twice for my subscription this month"}' \
  | python -m json.tool
```

```json
{
  "run_id": "c7a1...",
  "workflow_id": "customer_support",
  "status": "completed",
  "output": "I can see the duplicate charge on your account. I will issue a refund within 3-5 business days."
}
```

### Research Workflow — planner_executor_observer

The planner defines a research step, the executor carries it out, and the observer decides whether to continue or signal DONE.

```bash
curl -s -X POST http://localhost:8000/runs/ \
  -H "Content-Type: application/json" \
  -d '{"workflow_id": "research_workflow", "input": "Summarize recent advances in sparse attention mechanisms"}' \
  | python -m json.tool
```

```json
{
  "run_id": "d9e4...",
  "workflow_id": "research_workflow",
  "status": "completed",
  "output": "Found 4 key papers from 2023-2024 covering FlashAttention-2, Longformer, BigBird, and Mamba..."
}
```

### Polling a run

```bash
curl http://localhost:8000/runs/{run_id}
```

### HITL approve / reject

For workflows with `hitl_enabled: true` (not in the V1 demo set):

```bash
curl -X POST http://localhost:8000/runs/{run_id}/approve \
  -H "Content-Type: application/json" \
  -d '{"comment": "Looks good"}'

curl -X POST http://localhost:8000/runs/{run_id}/reject \
  -H "Content-Type: application/json" \
  -d '{"reason": "Output needs revision"}'
```

## Adding a New Workflow

1. Create `workflows/my_workflow/workflow.yaml`, `agents.yaml`, and `tools.yaml`
2. Restart the server — `ConfigLoader` picks it up automatically on startup
3. Call `POST /runs/` with `"workflow_id": "my_workflow"`

No platform code changes required.

## V2 Roadmap

- **Dynamic Workflow Builder** — compose workflows from natural language descriptions
- **Persistent storage** — PostgreSQL-backed RunManager and MemoryStore
- **HITL production support** — background task execution, `GET /hitl/pending` endpoint
- **MCP tool integration** — real Model Context Protocol server connections
- **Real-time event streaming** — WebSocket or SSE for live run progress
- **Multi-provider support** — Anthropic Claude, Google Gemini alongside OpenAI

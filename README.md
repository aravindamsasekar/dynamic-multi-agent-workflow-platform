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
  tools/                    # MockAdapter, HTTPAdapter, GitHubAdapter, MCPAdapter
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

# Required for the pr_review workflow (see GitHub Setup below)
GITHUB_TOKEN=github_pat_...
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

## GitHub Setup

The `pr_review` workflow calls the GitHub REST API. Configure authentication in `.env`:

```
GITHUB_TOKEN=github_pat_...
```

**Public repositories** — `GITHUB_TOKEN` is optional. Unauthenticated requests work but are rate-limited to 60 requests/hour per IP.

**Private repositories** — `GITHUB_TOKEN` is required. Create a fine-grained personal access token at GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens. Grant read-only access to:
- **Repository contents** — required to read diffs
- **Pull requests** — required to read PR metadata and file lists

Authenticated requests are rate-limited to 5,000 requests/hour.

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

### PR Review — parallel_specialist (GitHub integration)

Fetches PR metadata, changed files, and unified diff from GitHub, then generates a structured code review.

**Input contract** — pass `input` as a structured object (not a string):

| Field | Type | Description |
|---|---|---|
| `owner` | string | Repository owner or organization |
| `repo` | string | Repository name |
| `pull_number` | integer | Pull request number |

**curl:**

```bash
curl -s -X POST http://localhost:8000/runs/ \
  -H "Content-Type: application/json" \
  -d '{"workflow_id": "pr_review", "input": {"owner": "octocat", "repo": "Hello-World", "pull_number": 1}}' \
  | python -m json.tool
```

**PowerShell:**

```powershell
Invoke-RestMethod -Method POST `
  -Uri "http://localhost:8000/runs/" `
  -ContentType "application/json" `
  -Body '{"workflow_id":"pr_review","input":{"owner":"octocat","repo":"Hello-World","pull_number":1}}'
```

**Execution flow:**

```
POST /runs/  {"workflow_id": "pr_review", "input": {"owner": "octocat", "repo": "Hello-World", "pull_number": 1}}
  → Orchestrator serialises dict → JSON string; stores original dict in SharedState["workflow_input"]
  → ParallelSpecialistExecutor selected
  → github_fetch_agent receives JSON string as its user message
      → calls github_get_pr    → GET /repos/octocat/Hello-World/pulls/1
      → calls github_get_files → GET /repos/octocat/Hello-World/pulls/1/files
      → calls github_get_diff  → GET /repos/octocat/Hello-World/pulls/1  (Accept: application/vnd.github.diff)
      → returns structured summary of all three responses
  → review_agent receives the summary
      → returns structured review (Summary / Changes / Code Quality / Verdict)
  → RunManager marks run COMPLETED
```

**Expected response:**

```json
{
  "run_id": "...",
  "workflow_id": "pr_review",
  "status": "completed",
  "output": "## Summary\nThis PR adds...\n\n## Verdict\nApprove — ..."
}
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

## Knowledge Layer (RAG)

The platform includes a retrieval-augmented generation (RAG) layer that lets agents search internal knowledge bases before responding. The PR Review workflow uses this to ground reviews in team coding standards.

### How it works

1. **Source documents** live in `resources/knowledge/<collection>/` (plain text, Markdown, code).
2. At startup, `KnowledgeIndexer` scans each collection, computes SHA-256 hashes per file, and rebuilds the FAISS index only for collections where files have changed. Unchanged collections are skipped.
3. At runtime, `knowledge_search` is a normal tool. When an agent calls it, `KnowledgeAdapter` embeds the query with OpenAI, searches FAISS (cosine similarity), and returns the top-k matching chunks.
4. Generated artifacts (`data/knowledge/`) are gitignored.

### Configuration

Create `knowledge_config.yaml` in the project root:

```yaml
knowledge:
  embedding:
    model: text-embedding-3-small
  vector_store:
    path: data/knowledge
  chunking:
    size: 1000
    overlap: 200
  retrieval:
    top_k: 5
  collections:
    - name: coding-standards
      path: resources/knowledge/coding-standards
    - name: architecture
      path: resources/knowledge/architecture
    - name: runbooks
      path: resources/knowledge/runbooks
```

If `knowledge_config.yaml` is missing, the server starts normally and logs a warning. Knowledge tools in any workflow will return HTTP 503 until the file is created and the server is restarted.

### Adding knowledge documents

Place `.md`, `.txt`, `.py`, or any supported text file under the relevant collection directory:

```
resources/knowledge/
  coding-standards/
    coding_standards.md
    testing_guidelines.md
    pr_review_guidelines.md
  architecture/
    platform_architecture.md
  runbooks/
    pr_review_runbook.md
```

Then trigger re-indexing (see below).

### Building or rebuilding indexes

**Option A — Automatic at startup** (default): Every server start runs the indexer. Only changed collections are rebuilt.

**Option B — CLI script**:

```bash
python -m scripts.index_knowledge
```

Output:
```
  coding-standards: 42 chunk(s) indexed
  architecture: up to date (skipped)
  runbooks: 8 chunk(s) indexed

Done. 2/3 collection(s) rebuilt, 50 chunk(s) total.
```

### Testing the search API

```bash
curl -s -X POST http://localhost:8000/knowledge/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "missing tests in pull request",
    "collections": ["coding-standards"],
    "top_k": 3
  }' | python -m json.tool
```

List indexed collections:
```bash
curl http://localhost:8000/knowledge/collections
curl http://localhost:8000/knowledge/collections/coding-standards
```

### Running PR Review with RAG

The `pr_review` workflow is pre-configured to use `knowledge_search`. The `review_agent` will call it automatically before writing the final review.

**Requirements:**
- `OPENAI_API_KEY` set (for both LLM calls and embeddings)
- `GITHUB_TOKEN` set (for private repositories)
- `knowledge_config.yaml` present
- Knowledge base indexed (automatic at startup)

```bash
curl -s -X POST http://localhost:8000/runs/ \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_id": "pr_review",
    "input": {"owner": "octocat", "repo": "Hello-World", "pull_number": 42}
  }' | python -m json.tool
```

The review will include a "Standards Compliance" section citing specific retrieved guidelines.

## V2 Roadmap

- **Dynamic Workflow Builder** — compose workflows from natural language descriptions
- **Persistent storage** — PostgreSQL-backed RunManager and MemoryStore
- **HITL production support** — background task execution, `GET /hitl/pending` endpoint
- **MCP tool integration** — real Model Context Protocol server connections
- **Real-time event streaming** — WebSocket or SSE for live run progress
- **Multi-provider support** — Anthropic Claude, Google Gemini alongside OpenAI

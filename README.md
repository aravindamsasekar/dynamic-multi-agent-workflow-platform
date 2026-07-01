# Dynamic Multi-Agent Workflow Platform

A Python platform for building, running, and observing multi-agent AI workflows.
Run pre-built workflows from YAML, or describe your goal in plain English and let the
planner compose and execute a workflow on the fly — including generating new agents at
runtime when no static agent covers the required capability.

**1222 tests passing · Python 3.12+ · FastAPI · SQLite · FAISS · OpenAI**

---

## Why this exists

Most multi-agent demos hardcode agent logic and tool calls into application code.
This platform separates **what** agents do (YAML workflow definitions, or natural-language
goals) from **how** they do it (swappable LLM providers, adapter-based tools, pluggable
patterns). The result is a clean architectural boundary: new static workflows are YAML
files; new tool integrations are adapter classes; the orchestration engine never changes.
The planner adds a third path: describe the goal in English, and the platform builds the
workflow for you.

---

## Feature progression

| Version | What it adds |
|---|---|
| **V1** | Static workflow engine: YAML-driven workflows, 3 execution patterns, tool adapters (mock, HTTP), policy engine, HITL gate, FastAPI REST API |
| **V2** | Production integrations: GitHub adapter (PR data + diffs), FAISS knowledge/RAG layer, MCP adapter (stdio subprocess), SQLite persistence with full audit trail |
| **V3.1** | Natural-language planner: `GoalAnalyzer` → `PlanBuilder` → `PlanValidator`; capability-based agent/tool selection; HITL approval gate before execution; plan lifecycle (pending → executed / rejected) |
| **V3.2** | Runtime agent generation: planner generates `AgentDefinition` objects on-the-fly for capabilities with no static agent; dynamic tool binding at runtime; `FilesystemAdapter`; generatable-capability registry |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Two Entry Points                               │
│                                                                         │
│  Static path:  POST /runs/  ──────────────────────────────────────┐    │
│                                                                   ▼    │
│  Dynamic path: POST /planner/generate                       Orchestrator│
│                    │                                              │    │
│                    ▼                                              │    │
│             GoalAnalyzer (1 LLM call)                     Pattern Executor
│             capability allow-list filter              (ParallelSpecialist│
│                    │                                   Router · PEO)   │
│                    ▼                                              │    │
│             PlanBuilder (deterministic)                   AgentRuntime │
│             ├─ AgentSelector                         (LLM + tool loop) │
│             │   ├─ static agents from registry                    │    │
│             │   └─ generate agents at runtime ◄──────────────────┘    │
│             ├─ PatternSelector                                          │
│             └─ RuntimeAgentGenerator                                    │
│                    │                                                    │
│                    ▼                                                    │
│             PlanValidator (13 checks)          Tool Registry            │
│                    │                       ┌───────────────────────┐   │
│                    ▼                       │ GitHub · Knowledge    │   │
│              Plan (SQLite)                 │ MCP · HTTP · Mock     │   │
│              status: pending_review        │ Filesystem            │   │
│                    │                       └───────────────────────┘   │
│       ┌────────────┴──────────────┐                                    │
│       │ Human review & approval   │  ← GET /planner/{plan_id}         │
│       └────────────┬──────────────┘                                    │
│                    ▼                                                    │
│         POST /planner/{plan_id}/approve                                │
│              ExecutionAdapter                                           │
│              → WorkflowDefinition                                       │
│              → Orchestrator.run()  (V2 runtime, unchanged)             │
│                    │                                                    │
│                    ▼                                                    │
│             PersistingObserver ──► SQLite (runs · agents · tools · events)
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick start

### 1. Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

### 2. Configure environment

```powershell
Copy-Item .env.example .env
```

Edit `.env`:

```env
# Required for LLM calls and knowledge embeddings
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini          # optional, defaults to gpt-4o-mini

# Required for pr_review workflow (private repos) — omit for public repos
GITHUB_TOKEN=github_pat_...

# SQLite database path — defaults to ./workflow.db
DATABASE_URL=sqlite:///./workflow.db
```

### 3. Run tests

```powershell
python -m pytest tests/ -q
```

All 1222 tests use `MockLLMProvider` — no API keys required.

### 4. Start the API

```powershell
uvicorn api.main:app --reload
```

The server starts at `http://localhost:8000`. Interactive docs: `http://localhost:8000/docs`

At startup the server:
1. Loads all workflow YAML definitions from `workflows/`
2. Initializes the SQLite database
3. Indexes or refreshes knowledge collections (if `knowledge_config.yaml` is present)
4. Registers tool adapters (GitHub, Knowledge, MCP, Filesystem, Mock)
5. Registers generatable capabilities (e.g., `filesystem_read`) in the planner registry

---

## Project structure

```
api/
  main.py                    # FastAPI app + lifespan (startup indexing, shutdown)
  dependencies.py            # DI: registries, orchestrator, planner, tool wiring
  routers/                   # runs · workflows · knowledge · hitl · planner
  schemas/                   # Pydantic request/response models

platform/
  agent/                     # AgentRuntime — LLM + tool loop per agent
  aggregator/                # ResultAggregator (concatenate strategy)
  config/                    # ConfigLoader, ConfigValidator — parse workflow YAML
  core/
    exceptions.py
    interfaces/              # ILLMProvider · IToolAdapter · IObserver · IPolicyEngine
    models/                  # WorkflowDefinition · AgentDefinition · ExecutionContext …
  hitl/                      # ApprovalManager — pause/resume for human review
  knowledge/                 # KnowledgeService · KnowledgeIndexer · FAISS vector store
  llm/                       # OpenAIProvider · ClaudeProvider · MockLLMProvider
  memory/                    # InMemoryStore
  observability/             # ConsoleObserver · PersistingObserver
  orchestrator/              # Orchestrator · RunManager
  patterns/                  # ParallelSpecialistExecutor · RouterExecutor · PEOExecutor
  persistence/               # SQLAlchemy models + repositories (runs, agents, tools, events, plans)
  planner/
    goal_analyzer.py         # 1 LLM call → GoalAnalysis (capabilities + risk + HITL flag)
    plan_builder.py          # deterministic agent/tool/pattern selection + complexity estimate
    plan_validator.py        # 13 validation checks → ValidationResult (errors + warnings)
    planner_service.py       # orchestrates GoalAnalyzer → PlanBuilder → PlanValidator
    agent_selector.py        # matches required capabilities to static agents + generates new ones
    runtime_agent_generator.py # synthesizes AgentDefinition for capabilities with no static agent
    pattern_selector.py      # selects execution pattern from trigger_capabilities
    task_label_inferer.py    # infers human-readable label from required capabilities
    capability_registry.py   # static agent/tool descriptors + generatable capability registry
    execution_adapter.py     # GeneratedWorkflowPlan → WorkflowDefinition → Orchestrator.run()
    models.py                # GeneratedWorkflowPlan · GoalAnalysis · ValidationResult · …
    serialization.py         # JSON round-trip for plans and validation results
  policy/                    # PolicyEngine + ContentFilterRule
  registries/                # WorkflowRegistry · AgentRegistry · ToolRegistry
  state/                     # SharedState — per-run cross-agent key-value store
  tools/
    github_adapter.py        # GitHub REST API (PR metadata, files, diffs)
    knowledge_adapter.py     # FAISS semantic search
    mcp_adapter.py           # Any MCP server via stdio subprocess
    mcp_connection_manager.py
    filesystem_adapter.py    # Local file reads (V3.2)
    http_adapter.py          # Generic HTTP GET/POST
    mock_adapter.py          # Static responses for tests

workflows/                   # YAML workflow definitions (one folder per workflow)
  pr_review/                 # 4-agent production PR review (GitHub + RAG + MCP)
  devops_remediation/        # MCP filesystem analysis
  incident_commander/        # Parallel incident triage
  customer_support/          # Router-based support routing
  research_workflow/         # Planner-executor-observer research loop

resources/
  knowledge/
    coding-standards/        # PR review guidelines · coding standards · testing · security
    architecture/            # Platform architecture docs
    runbooks/                # Operational runbooks

tests/
  unit/                      # Fast isolated tests — no real API calls
  integration/               # End-to-end pattern tests using real YAML + MockLLMProvider
```

---

## API reference

### Static workflow execution

| Method | Path | Description |
|---|---|---|
| `POST` | `/runs/` | Trigger a workflow run |
| `GET` | `/runs/` | List all historical runs |
| `GET` | `/runs/{run_id}` | Get run status and output |
| `GET` | `/runs/{run_id}/details` | Full run details: agents, tool calls, events |
| `GET` | `/runs/{run_id}/events` | Raw event stream for a run |
| `POST` | `/runs/{run_id}/approve` | Approve a paused HITL run |
| `POST` | `/runs/{run_id}/reject` | Reject a paused HITL run |
| `GET` | `/workflows/` | List all loaded workflow definitions |

### Dynamic workflow planner

| Method | Path | Description |
|---|---|---|
| `POST` | `/planner/generate` | Analyze a natural-language goal, build and validate a plan (HTTP 201) |
| `GET` | `/planner/{plan_id}` | Retrieve a previously generated plan by ID |
| `POST` | `/planner/{plan_id}/approve` | Approve a valid plan and execute it; returns run result |
| `POST` | `/planner/{plan_id}/reject` | Reject a plan so it will not be executed |

**`/planner/{plan_id}/approve` status codes**

| Code | Condition |
|---|---|
| `200` | Plan approved and executed successfully |
| `404` | Plan not found |
| `409` | Plan not in `pending_review` status (already executed / rejected) OR validation failed |

### Knowledge

| Method | Path | Description |
|---|---|---|
| `POST` | `/knowledge/search` | Search knowledge collections |
| `GET` | `/knowledge/collections` | List indexed collections with stats |
| `GET` | `/knowledge/collections/{name}` | Collection detail (documents, chunk count) |

---

## Demo scenarios

All examples use PowerShell. `curl` equivalents follow each block.

---

### Scenario A — Static PR Review

Run the pre-built 4-agent PR review workflow directly without a planner.

```powershell
$body = @{
    workflow_id = "pr_review"
    input = @{ owner = "octocat"; repo = "Hello-World"; pull_number = 1 }
} | ConvertTo-Json -Depth 3

$run = Invoke-RestMethod -Method POST `
    -Uri "http://localhost:8000/runs/" `
    -ContentType "application/json" `
    -Body $body

$run | ConvertTo-Json -Depth 3
```

```bash
curl -s -X POST http://localhost:8000/runs/ \
  -H "Content-Type: application/json" \
  -d '{"workflow_id":"pr_review","input":{"owner":"octocat","repo":"Hello-World","pull_number":1}}' \
  | python -m json.tool
```

**What happens:**

```
POST /runs/  {workflow_id: pr_review, input: {owner, repo, pull_number}}
  → Orchestrator → ParallelSpecialistExecutor

  Parallel (3 agents run concurrently):
    pr_data_agent
      → github_get_pr    GET /repos/{owner}/{repo}/pulls/{n}
      → github_get_files GET /repos/{owner}/{repo}/pulls/{n}/files
      → github_get_diff  GET /repos/{owner}/{repo}/pulls/{n}  (diff media type)
      → returns structured PR data summary

    review_specialist
      → github_get_diff
      → knowledge_search (coding-standards + architecture collections)
      → returns: Code Quality · Architecture · Maintainability · Standards Compliance

    risk_specialist
      → github_get_diff
      → knowledge_search (security + testing guidelines)
      → returns: Security · Testing · Reliability · Performance

  Sequential (after all parallel complete):
    synthesis_agent
      → [optional] mcp_get_pr_comments (reads prior review threads via GitHub MCP)
      → returns: structured report with Verdict (APPROVED / REQUEST CHANGES / COMMENT)

  → PersistingObserver writes run, agent results, tool calls, events to SQLite
```

**Expected response shape:**

```json
{
  "run_id": "3f7a...",
  "workflow_id": "pr_review",
  "status": "completed",
  "output": "## Pull Request Summary\n...\n## Verdict\nAPPROVED — ..."
}
```

**Inspect the full audit trail:**

```powershell
Invoke-RestMethod "http://localhost:8000/runs/$($run.run_id)/details" | ConvertTo-Json -Depth 5
```

---

### Scenario B — Unsupported Goal (409)

Goals that require capabilities with no registered agents or generatable tools
produce an invalid plan. Attempting to approve an invalid plan returns HTTP 409.

**Step 1 — Generate a plan for an unsupported goal**

```powershell
$body = @{ goal = "Deploy a Kubernetes cluster and spin up 50 microservices" } | ConvertTo-Json
$plan = Invoke-RestMethod -Method POST `
    -Uri "http://localhost:8000/planner/generate" `
    -ContentType "application/json" `
    -Body $body
$plan | ConvertTo-Json -Depth 5
$planId = $plan.plan_id
```

```bash
PLAN=$(curl -s -X POST http://localhost:8000/planner/generate \
  -H "Content-Type: application/json" \
  -d '{"goal":"Deploy a Kubernetes cluster and spin up 50 microservices"}')
echo $PLAN | python -m json.tool
PLAN_ID=$(echo $PLAN | python -c "import sys,json; print(json.load(sys.stdin)['plan_id'])")
```

**Plan response highlights:**

```json
{
  "plan_id": "b9c2d4e1-...",
  "goal": "Deploy a Kubernetes cluster and spin up 50 microservices",
  "status": "pending_review",
  "executable": false,
  "validation": {
    "is_valid": false,
    "errors": [
      {
        "code": "MISSING_CAPABILITIES",
        "message": "Required capabilities have no matching agents or tools: deploy_kubernetes, manage_microservices"
      }
    ],
    "warnings": []
  }
}
```

The `executable: false` field signals the plan cannot be approved. The plan is still
persisted so you can inspect the validation errors.

**Step 2 — Try to approve the invalid plan (expect 409)**

```powershell
try {
    Invoke-RestMethod -Method POST `
        -Uri "http://localhost:8000/planner/$planId/approve" `
        -ContentType "application/json" `
        -Body '{}'
} catch {
    $_.Exception.Response.StatusCode   # 409
    $_.ErrorDetails.Message            # "Plan cannot be approved: validation failed"
}
```

```bash
curl -s -o /dev/null -w "%{http_code}" \
  -X POST "http://localhost:8000/planner/$PLAN_ID/approve" \
  -H "Content-Type: application/json" \
  -d '{}'
# prints: 409
```

**Why 409?** The approve endpoint checks `validation.is_valid` before executing. A plan with
missing capabilities is persisted (so you can inspect the errors) but blocked from execution.

---

### Scenario C — Generated Runtime Agent (Read README.md)

When a goal requires a capability with no static agent, the planner generates an
`AgentDefinition` at runtime and binds the appropriate tool. The generated agent is
registered temporarily for the run, then cleaned up.

**Step 1 — Generate the plan**

```powershell
$body = @{ goal = "Read README.md" } | ConvertTo-Json
$plan = Invoke-RestMethod -Method POST `
    -Uri "http://localhost:8000/planner/generate" `
    -ContentType "application/json" `
    -Body $body
$plan | ConvertTo-Json -Depth 5
$planId = $plan.plan_id
```

```bash
PLAN=$(curl -s -X POST http://localhost:8000/planner/generate \
  -H "Content-Type: application/json" \
  -d '{"goal":"Read README.md"}')
echo $PLAN | python -m json.tool
PLAN_ID=$(echo $PLAN | python -c "import sys,json; print(json.load(sys.stdin)['plan_id'])")
```

**Plan response highlights:**

```json
{
  "plan_id": "c7e3f1a2-...",
  "goal": "Read README.md",
  "status": "pending_review",
  "executable": true,
  "task_label": "filesystem_read",
  "selected_pattern": "parallel_specialist",
  "runtime_agents": [
    {
      "id": "gen_c7e3f1a2_filesystem_read",
      "name": "Filesystem Read Agent",
      "capabilities": ["filesystem_read"],
      "tool_names": ["filesystem_read_file"],
      "generated": true
    }
  ],
  "validation": { "is_valid": true, "errors": [], "warnings": [] }
}
```

`generated: true` — this agent does not exist in the static registry. It is synthesized by
`RuntimeAgentGenerator` and bound to `filesystem_read_file` (a `FilesystemAdapter` tool).

**Step 2 — Review the plan**

```powershell
Invoke-RestMethod "http://localhost:8000/planner/$planId" | ConvertTo-Json -Depth 5
```

**Step 3 — Approve and execute**

```powershell
$result = Invoke-RestMethod -Method POST `
    -Uri "http://localhost:8000/planner/$planId/approve" `
    -ContentType "application/json" `
    -Body '{}'
$result | ConvertTo-Json -Depth 3
```

```bash
curl -s -X POST "http://localhost:8000/planner/$PLAN_ID/approve" \
  -H "Content-Type: application/json" \
  -d '{}' | python -m json.tool
```

**What happens:**

```
POST /planner/{plan_id}/approve  (input_data: {})
  → ExecutionAdapter.execute()
    effective_input = plan.user_goal = "Read README.md"   ← goal fallback: empty input_data
    → registers gen_{plan_id}_filesystem_read in AgentRegistry (temporary)
    → Orchestrator.run()
        → ParallelSpecialistExecutor (no reviewer — no synthesize_findings capability)
            → AgentRuntime: task = "Read README.md"
               → calls filesystem_read_file(path="README.md")
               → FilesystemAdapter → Path("README.md").read_text()
               → returns file contents
  → unregisters generated agent (try/finally cleanup)
```

**Expected response:**

```json
{
  "plan_id": "c7e3f1a2-...",
  "run_id": "9a1b3c5d-...",
  "status": "completed",
  "output": "# Dynamic Multi-Agent Workflow Platform\n\nA Python platform..."
}
```

**Step 4 — Inspect the run**

```powershell
$runId = $result.run_id
Invoke-RestMethod "http://localhost:8000/runs/$runId/details" | ConvertTo-Json -Depth 5
```

The `/details` response shows the generated agent ID, the `filesystem_read_file` tool call
with `input: {"path": "README.md"}`, and the file contents as `output`.

---

## Execution patterns

| Pattern | `pattern` key | Description |
|---|---|---|
| Parallel Specialist | `parallel_specialist` | N agents run concurrently; outputs concatenated; optional reviewer synthesizes |
| Router | `router` | Classifier agent selects a route label; matched specialist handles the request |
| Planner-Executor-Observer | `planner_executor_observer` | Iterative loop: planner → executor → observer signals DONE or RETRY |

---

## Tool adapters

| Adapter | Class | Purpose |
|---|---|---|
| `github` | `GitHubAdapter` | GitHub REST API — fetch PRs, files, diffs |
| `knowledge` | `KnowledgeAdapter` | FAISS semantic search over indexed Markdown/text |
| `mcp` | `MCPAdapter` | Any MCP server via stdio subprocess |
| `http` | `HTTPAdapter` | Generic HTTP GET/POST with configurable headers |
| `filesystem` | `FilesystemAdapter` | Local file reads — used by generated agents (V3.2) |
| `mock` | `MockAdapter` | Static responses for local development and testing |

---

## Included static workflows

| Workflow ID | Pattern | Description |
|---|---|---|
| `pr_review` | parallel_specialist | 4-agent PR review: data fetch → code review + risk assessment → synthesis |
| `devops_remediation` | parallel_specialist | Read files via MCP filesystem server, produce structured analysis |
| `incident_commander` | parallel_specialist | Parallel triage: metrics + logs + deployment → reviewer synthesizes root cause |
| `customer_support` | router | Classify billing vs. technical query, route to specialist |
| `research_workflow` | planner_executor_observer | Iterative research loop with observer-controlled termination |

---

## Dynamic workflow planner (V3.1 / V3.2)

### How it works

```
POST /planner/generate  {"goal": "Review PR #42 in org/repo"}
  ↓
GoalAnalyzer (1 LLM call)
  → extracts required_capabilities from the goal text
  → filters against all_capabilities() allow-list (anti-hallucination)
  → assesses risk_level + requires_hitl
  ↓
TaskLabelInferer (deterministic)
  → derives human-readable task_label from required_capabilities
  ↓
PlanBuilder (deterministic Python)
  → AgentSelector
      ├─ matches caps to static agents from CapabilityRegistry
      └─ calls RuntimeAgentGenerator for caps with no static agent
  → PatternSelector (picks pattern from trigger_capabilities)
  → selects tools, generates guardrails, estimates complexity + duration
  ↓
PlanValidator (deterministic Python)
  → NO_REQUIRED_CAPABILITIES — short-circuit if goal produced no capabilities
  → MISSING_CAPABILITIES — required caps not covered by any agent or tool (blocks approve)
  → DATAFLOW_UNSATISFIED — agent consumes tokens not produced by selected agents
  → 10 additional checks (agent conflicts, tool availability, …)
  ↓
Plan persisted to SQLite (status: pending_review, executable: is_valid)
```

### Runtime agent generation (V3.2)

When `AgentSelector` encounters a required capability with no matching static agent,
`RuntimeAgentGenerator` synthesizes a complete `AgentDefinition`:

- **Agent ID**: `gen_{plan_id}_{capability}` (e.g., `gen_c7e3_filesystem_read`)
- **System prompt**: generated from the capability name and tool descriptions
- **Tool binding**: tools registered in `CapabilityRegistry` for that capability

Generated agents are **temporarily registered** for the run and **cleaned up via
`try/finally`** in `ExecutionAdapter` — they never persist in the agent registry.

### Registering a new generatable capability

```python
# In api/dependencies.py
tl_registry.register(
    "my_tool",
    MyAdapter(),
    ToolDefinition(
        name="my_tool",
        description="...",
        input_schema={"type": "object", "properties": {"param": {"type": "string"}}, "required": ["param"]},
        adapter_type=AdapterType.MY_TYPE,
    ),
)
cap_registry.register_tool(ToolCapabilityDescriptor(
    tool_name="my_tool",
    name="My Tool",
    description="...",
    capabilities=["my_capability"],
    operation_type=OperationType.READ,
    data_source="local",
))
cap_registry.register_generatable_capability("my_capability")
```

`register_generatable_capability` adds `my_capability` to the `GoalAnalyzer` allow-list
without exposing all tool-plumbing capabilities to the LLM.

### Plan lifecycle

```
pending_review  ──approve (valid)──►   executed
                ──approve (invalid)──► 409 (status unchanged)
                ──reject──►            rejected
```

### Planner API quick reference

#### POST /planner/generate

Request: `{ "goal": "Review pull request #42 in the org/myrepo repository" }`

Response `201`:
```json
{
  "plan_id": "a3f7b2c1-...",
  "goal": "...",
  "status": "pending_review",
  "executable": true,
  "task_label": "code_review",
  "goal_analysis": {
    "required_capabilities": ["fetch_pr_data", "review_code_quality", "assess_security", "synthesize_findings"],
    "risk_level": "low",
    "confidence": 0.92,
    "reasoning": "...",
    "constraints": ["read_only"],
    "requires_hitl": false
  },
  "selected_pattern": "parallel_specialist",
  "selected_agents": ["pr_data_agent", "review_specialist", "risk_specialist", "synthesis_agent"],
  "runtime_agents": [...],
  "validation": { "is_valid": true, "errors": [], "warnings": [] }
}
```

Error `422`: LLM response could not be parsed (prompt injection, network error, malformed response).

#### POST /planner/{plan_id}/approve

Request: `{ "input_data": { "owner": "org", "repo": "myrepo", "pull_number": 42 } }`

`input_data` can be a string or dict. If empty (`{}`), the plan's `user_goal` is used as
the workflow input — generated agents receive the original natural-language goal.

Response `200`:
```json
{
  "plan_id": "a3f7b2c1-...",
  "run_id": "8b2e4f1a-...",
  "status": "completed",
  "output": "## Pull Request Review\n..."
}
```

---

## GitHub integration

The `github` adapter calls the GitHub REST API via `httpx`.

| Tool name | Operation | GitHub endpoint |
|---|---|---|
| `github_get_pr` | `get_pull_request` | `GET /repos/{owner}/{repo}/pulls/{n}` |
| `github_get_files` | `get_changed_files` | `GET /repos/{owner}/{repo}/pulls/{n}/files` |
| `github_get_diff` | `get_diff` | `GET /repos/{owner}/{repo}/pulls/{n}` + diff media type |

Authentication: `GITHUB_TOKEN` env var. Optional for public repos (60 req/hr unauthenticated; 5,000/hr with token).

**Create a fine-grained token:** GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens. Grant read-only access to **Repository contents** and **Pull requests**.

---

## Knowledge / RAG layer

```
resources/knowledge/
  coding-standards/   ← coding standards, testing, security, PR review guidelines
  architecture/       ← platform architecture reference
  runbooks/           ← operational runbooks
```

**How it works:**

1. At startup, `KnowledgeIndexer` hashes every source file in each collection directory.
2. Only collections whose files have changed (or are new) are re-indexed.
3. Documents are chunked, embedded with OpenAI `text-embedding-3-small`, and stored in FAISS backed by SQLite metadata.
4. At runtime, `knowledge_search` embeds the query, runs cosine-similarity search, and returns top-k chunks with source file and score.

**Configuration** (`knowledge_config.yaml`):

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

If `knowledge_config.yaml` is missing, the server starts and logs a warning. Knowledge endpoints return HTTP 503.

**Adding new documents:** drop any `.md` or `.txt` file into the relevant collection directory and restart the server.

---

## MCP integration

The `mcp` adapter connects to any [Model Context Protocol](https://modelcontextprotocol.io) server via stdio subprocess.

**How `MCPConnectionManager` works:**

1. On first tool call, the manager starts the MCP server subprocess via `npx` (or any configured command).
2. A `ClientSession` is created and initialized. `list_tools()` discovers available tools.
3. The session is **reused** for all subsequent calls — no reconnect overhead per tool call.
4. If the transport dies, the manager reconnects automatically on the next call.
5. At server shutdown, `adapter.close()` cleanly exits the session and kills the subprocess.

**Example configuration** (`tools.yaml`):

```yaml
- name: filesystem_read_file
  adapter_type: mcp
  adapter_config:
    server_command: npx
    server_args: ["-y", "@modelcontextprotocol/server-filesystem", "."]
    tool_name: read_file
```

**MCP servers used by included workflows:**

| Workflow | MCP server | Tool |
|---|---|---|
| `devops_remediation` | `@modelcontextprotocol/server-filesystem` | `read_file` |
| `pr_review` (synthesis) | `@modelcontextprotocol/server-github` | `list_pull_request_review_comments` |

**Requirements:** Node.js must be installed for `npx` to work.

---

## SQLite persistence

Every workflow run is automatically persisted to `workflow.db`.

| Table | Contents |
|---|---|
| `workflow_runs` | run_id, workflow_id, status, input, output, error, timestamps |
| `agent_results` | per-agent output for each run |
| `tool_calls` | every tool invoked: name, input JSON, output text, is_error flag |
| `workflow_events` | ordered event stream: run_started, agent_started, tool_called, run_completed … |
| `generated_plans` | plan_id, goal, plan JSON, validation JSON, status, execution_run_id |

**Query run history:**

```powershell
Invoke-RestMethod "http://localhost:8000/runs/"
Invoke-RestMethod "http://localhost:8000/runs/{run_id}/details"
Invoke-RestMethod "http://localhost:8000/runs/{run_id}/events"
```

---

## HITL (Human-in-the-Loop)

**Static workflows:** workflows with `hitl_enabled: true` pause after the parallel specialist
phase and wait for manual approval before the reviewer agent runs.

**Dynamic plans:** `GoalAnalyzer` sets `requires_hitl: true` for goals classified as high-risk.
`ExecutionAdapter` propagates this to the generated `WorkflowDefinition`.

```bash
# Approve a paused run
curl -X POST http://localhost:8000/runs/{run_id}/approve \
  -H "Content-Type: application/json" \
  -d '{"comment": "Looks good, proceed"}'

# Reject a paused run
curl -X POST http://localhost:8000/runs/{run_id}/reject \
  -H "Content-Type: application/json" \
  -d '{"reason": "Specialist outputs need revision"}'
```

---

## Adding a new static workflow

1. Create `workflows/my_workflow/workflow.yaml`, `agents.yaml`, and `tools.yaml`
2. Restart the server — `ConfigLoader` scans `workflows/` at startup and registers everything automatically
3. `POST /runs/` with `"workflow_id": "my_workflow"`

No platform code changes required.

**Minimal workflow.yaml:**

```yaml
workflow_id: my_workflow
name: My Workflow
pattern: parallel_specialist
agent_ids: [specialist_a, specialist_b]
pattern_config:
  strategy: concatenate
  reviewer_agent_id: reviewer_agent
hitl_enabled: false
```

---

## Roadmap

| Feature | Status | Description |
|---|---|---|
| Static workflow engine (V1) | **Complete** | 3 patterns, YAML workflows, FastAPI, HITL, mock/HTTP adapters |
| Production integrations (V2) | **Complete** | GitHub, FAISS/RAG, MCP, SQLite persistence |
| Dynamic workflow generation (V3.1) | **Complete** | Natural-language planner: GoalAnalyzer → PlanBuilder → PlanValidator |
| Runtime agent generation (V3.2) | **Complete** | RuntimeAgentGenerator, FilesystemAdapter, generatable capability registry |
| Multi-generated-agent workflows | Planned | Plans that compose multiple generated agents (e.g., fetch + summarize + write) |
| Expanded generatable capabilities | Planned | HTTP fetch, SQL query, shell exec as generatable tool-backed capabilities |
| MCP tool discovery in planner | Planned | `register_generatable_capability` auto-populated from MCP `list_tools()` at startup |
| PostgreSQL backend | Planned | Replace SQLite with PostgreSQL for production multi-instance deployments |
| Background task execution | Planned | Non-blocking `POST /runs/` with SSE/WebSocket for live progress streaming |
| Multi-provider LLM | Planned | Anthropic Claude and Google Gemini alongside OpenAI, runtime-selectable per agent |
| Agent memory persistence | Planned | Long-term per-agent memory across runs using the `IMemoryStore` interface |
| Visual workflow editor | Planned | Browser-based graph editor that generates YAML and submits plans |
| Auth and multi-tenancy | Planned | API key authentication and tenant-scoped workflow isolation |

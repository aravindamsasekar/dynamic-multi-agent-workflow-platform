# V2 Design: Real-World Grounding

**Status:** Planned  
**Target start:** 2026-07-01  
**Target complete:** 2026-07-28 (~4 weeks)

---

## Context

V1 delivered a complete, test-covered execution engine: three patterns (parallel specialist, router, planner-executor-observer), YAML-driven workflow configuration, a FastAPI layer, and 217 passing tests. All storage is in-memory; all tool calls use mock or HTTP adapters; there is no retrieval or real external integrations.

V2 makes the engine useful for real engineering workflows by grounding it in three concrete capabilities: a GitHub tool adapter, SQLite persistence, and runbook retrieval (RAG). These are combined into one end-to-end demo workflow.

V3, which follows, is dynamic workflow generation — the platform composes workflows from natural language descriptions at runtime.

---

## Goals

1. **GitHub tool adapter** — agents can call GitHub API operations (list PRs, read file contents, post comments) through the existing `IToolAdapter` interface.
2. **SQLite persistence** — `WorkflowRun` records and per-agent conversation history survive server restarts; existing `IMemoryStore` and run-storage contracts are satisfied by SQLite-backed implementations.
3. **Runbook retrieval (RAG)** — a new `IRetriever` interface backs a `RunbookRetriever` that embeds Markdown documents into SQLite, then returns the top-k semantically relevant chunks at query time to enrich an agent's context.
4. **PR triage workflow** — one end-to-end workflow (`pr_triage`) that exercises all three pieces: fetch open PRs from GitHub, retrieve relevant engineering standards from the runbook index, run three specialist agents in parallel (risk, coverage, style), and emit a triage recommendation.

---

## Non-Goals

- **No dynamic workflow generation** — that is V3. All workflows are still defined as YAML files.
- **No PostgreSQL / production-grade DB** — SQLite is sufficient for a local demo. Migrating to PostgreSQL is a V3+ concern.
- **No vector-DB service** — vectors are stored as JSON blobs in SQLite and similarity is computed in Python. No Pinecone, Weaviate, or pgvector.
- **No streaming / WebSocket** — `POST /runs/` stays synchronous in V2. Background execution and SSE are V3.
- **No multi-LLM-provider expansion** — the Claude provider already exists in `platform/llm/claude_provider.py`. V2 does not add Gemini or expand provider coverage further.
- **No auth / multi-tenancy** — the API remains unauthenticated and single-tenant.
- **No changes to V1 patterns** — parallel specialist, router, and planner-executor-observer are untouched.

---

## Architecture Changes

### New packages

```
platform/
  storage/               # SQLite-backed persistence
    __init__.py
    sqlite_run_store.py  # WorkflowRun CRUD
    sqlite_memory_store.py  # IMemoryStore over SQLite
    schema.py            # CREATE TABLE statements + migrations
  retrieval/             # RAG
    __init__.py
    interface.py         # IRetriever
    runbook_retriever.py # OpenAI embeddings + SQLite vector search
    ingest.py            # CLI: python -m platform.retrieval.ingest --source <dir>
  tools/
    github_adapter.py    # IToolAdapter → GitHub REST API
```

### Interface additions

**`platform/core/interfaces/retriever.py`** (new)

```python
class IRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        ...

@dataclass
class RetrievedChunk:
    source: str      # file path or document id
    content: str     # raw text chunk
    score: float     # cosine similarity [0, 1]
```

This interface is in `core/interfaces/` so it is visible to pattern executors and the orchestrator without touching the `api/` layer.

### Changes to existing components

| Component | Change |
|---|---|
| `RunManager` | Accept optional `IRunStore` constructor parameter. Default to `InMemoryRunStore` (extracted from the existing dict). `SQLiteRunStore` implements the same interface. |
| `ConfigLoader._ADAPTER_BUILDERS` | Add `"github"` entry mapping to `GitHubAdapter` factory. |
| `WorkflowDefinition` | Add optional `retriever_config: dict | None = None`. When set, the orchestrator instantiates a retriever and injects retrieved context into the first agent's system prompt before execution. |
| `AdapterType` enum | Add `GITHUB = "github"`. |
| `api/main.py` | Accept `DB_PATH` env var (default `runs.db`) and pass `SQLiteRunStore` + `SQLiteMemoryStore` into the orchestrator at startup. |

### SQLite schema

```sql
-- runs.db

CREATE TABLE IF NOT EXISTS workflow_runs (
    run_id      TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    status      TEXT NOT NULL,
    input       TEXT NOT NULL,
    output      TEXT,
    error       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    agent_id    TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runbook_chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   TEXT NOT NULL   -- JSON array of floats
);

CREATE INDEX IF NOT EXISTS idx_agent_messages_run_agent
    ON agent_messages (run_id, agent_id);
```

### GitHub adapter

Configured in `tools.yaml` as:

```yaml
tools:
  - tool_id: github_list_prs
    adapter_type: github
    config:
      operation: list_prs
      owner: "${GITHUB_OWNER}"
      repo: "${GITHUB_REPO}"

  - tool_id: github_get_file
    adapter_type: github
    config:
      operation: get_file_contents
      owner: "${GITHUB_OWNER}"
      repo: "${GITHUB_REPO}"
```

`GitHubAdapter` reads `GITHUB_TOKEN` from the environment at construction time and raises `RuntimeError` with a clear message if it is not set. It uses `httpx.AsyncClient` for all calls (consistent with `HTTPAdapter`). Supported operations in V2:

| Operation | GitHub endpoint |
|---|---|
| `list_prs` | `GET /repos/{owner}/{repo}/pulls` |
| `get_pr_details` | `GET /repos/{owner}/{repo}/pulls/{number}` |
| `get_file_contents` | `GET /repos/{owner}/{repo}/contents/{path}` |
| `create_pr_comment` | `POST /repos/{owner}/{repo}/issues/{number}/comments` |
| `list_issues` | `GET /repos/{owner}/{repo}/issues` |

### RAG flow

**Ingestion** (run once, or on a schedule):

```bash
python -m platform.retrieval.ingest \
    --source docs/runbooks/ \
    --db runs.db \
    --chunk-size 500
```

`ingest.py` walks the source directory for `.md` and `.txt` files, splits them into `chunk-size`-token chunks with 50-token overlap, calls the OpenAI embeddings API (`text-embedding-3-small`), and writes rows to `runbook_chunks`.

**Retrieval at runtime:**

When `retriever_config` is present on a `WorkflowDefinition`, the orchestrator calls `retriever.retrieve(workflow_input, top_k=5)` before dispatching to the first pattern executor. The top-k chunks are prepended to the first agent's system prompt under a `## Relevant Context` header:

```
## Relevant Context
[from docs/runbooks/pr_standards.md, score=0.91]
All PRs touching the payments service must include a rollback plan...

[from docs/runbooks/coverage_policy.md, score=0.87]
Minimum test coverage for new code is 80%...
```

No new pattern type is introduced — retrieval augmentation is an orchestrator-level pre-processing step applied transparently before the chosen pattern executes.

---

## End-to-End Workflow: `pr_triage`

Uses the `parallel_specialist` pattern. Full YAML lives in `workflows/pr_triage/`.

```
Input: "Triage open PRs for repo owner/repo"
          │
          ▼
   [GitHub: list_prs]          ← GitHubAdapter
          │
          ▼
   [RunbookRetriever]           ← top-5 chunks injected into system prompt
          │
          ▼
   ┌──────┬──────┬──────┐
   │ Risk │ Cov. │Style │      ← parallel specialist agents
   └──────┴──────┴──────┘
          │
          ▼
      [Reviewer]               ← aggregates + emits triage recommendation
          │
          ▼
   Output: ranked triage list + rationale
```

This workflow is the acceptance test for V2: if it runs end-to-end against a real GitHub repo with a real API key, all four V2 capabilities are exercised simultaneously.

---

## Phases

### Phase 1 — SQLite persistence (Week 1)

- `platform/storage/schema.py`: schema constants and `init_db(db_path)`.
- `platform/storage/sqlite_run_store.py`: `IRunStore` + `SQLiteRunStore`.
- `platform/storage/sqlite_memory_store.py`: `IMemoryStore` + `SQLiteMemoryStore`.
- Extract `InMemoryRunStore` from `RunManager` so it satisfies `IRunStore` and the default path still works.
- Update `RunManager.__init__` to accept `run_store: IRunStore`.
- Update `api/main.py` to wire `SQLiteRunStore` + `SQLiteMemoryStore` when `DB_PATH` is set.
- Unit tests for both SQLite implementations (tmp file, no mocking of SQLite itself).

**Deliverable:** Server can restart and `GET /runs/{run_id}` still returns completed run records.

---

### Phase 2 — GitHub tool adapter (Week 2)

- `platform/tools/github_adapter.py`: `GitHubAdapter(IToolAdapter)`.
- Add `GITHUB = "github"` to `AdapterType` enum in `platform/core/models/tool.py`.
- Add `"github"` entry to `ConfigLoader._ADAPTER_BUILDERS`.
- Unit tests using `respx` (or `unittest.mock`) to mock `httpx` calls; no real network calls in CI.

**Deliverable:** A workflow YAML can specify `adapter_type: github` and an agent will call the GitHub API.

---

### Phase 3 — Runbook retrieval (Week 3)

- `platform/core/interfaces/retriever.py`: `IRetriever`, `RetrievedChunk`.
- `platform/retrieval/runbook_retriever.py`: `RunbookRetriever(IRetriever)` using OpenAI embeddings + cosine similarity over SQLite rows.
- `platform/retrieval/ingest.py`: CLI ingestion script.
- Add `retriever_config` field to `WorkflowDefinition`.
- Update `Orchestrator.run()` to instantiate the retriever and inject context when `retriever_config` is set.
- Unit tests: use a small fixture with pre-computed embeddings stored in the test assets; no live OpenAI calls.

**Deliverable:** `python -m platform.retrieval.ingest --source docs/runbooks/ --db runs.db` populates the DB; a workflow with `retriever_config` gets enriched prompts.

---

### Phase 4 — PR triage workflow and integration test (Week 4)

- `workflows/pr_triage/workflow.yaml`, `agents.yaml`, `tools.yaml`.
- `docs/runbooks/` — seed runbook files covering PR standards, coverage policy, security review checklist.
- Integration test in `tests/integration/test_pr_triage.py` using `MockLLMProvider` + mocked GitHub HTTP calls (no live credentials required).
- Manual smoke test against a real GitHub repo documented in `docs/v2-smoke-test.md`.

**Deliverable:** `pytest tests/integration/test_pr_triage.py` passes; manual smoke test confirmed working.

---

## Testing Strategy

### Unit tests

Each new component gets a dedicated unit test file that does not touch the network or a live LLM:

| New component | Test file |
|---|---|
| `SQLiteRunStore` | `tests/unit/storage/test_sqlite_run_store.py` |
| `SQLiteMemoryStore` | `tests/unit/storage/test_sqlite_memory_store.py` |
| `GitHubAdapter` | `tests/unit/tools/test_github_adapter.py` (httpx mocked with `respx`) |
| `RunbookRetriever` | `tests/unit/retrieval/test_runbook_retriever.py` (pre-computed fixture embeddings) |
| `ingest.py` | `tests/unit/retrieval/test_ingest.py` (tmp dir, fixture docs) |

### Integration tests

- `tests/integration/test_pr_triage.py` — full workflow execution using `MockLLMProvider` + mocked GitHub responses. Validates that the orchestrator wires retriever context injection, GitHub calls, parallel specialist execution, and SQLite persistence in sequence.
- Existing 217 V1 integration tests must continue to pass without modification.

### Regression

V1 tests run with `InMemoryRunStore` and `InMemoryStore` defaults — no changes required. The SQLite path is activated only when `DB_PATH` is set, so V1 test infrastructure is unaffected.

---

## Success Criteria

V2 is complete when all of the following are true:

1. All 217 V1 tests pass without modification.
2. New unit tests pass for `SQLiteRunStore`, `SQLiteMemoryStore`, `GitHubAdapter`, and `RunbookRetriever`.
3. `tests/integration/test_pr_triage.py` passes against mock data.
4. Server restart does not lose run history: `GET /runs/{run_id}` returns completed records after restart when `DB_PATH` is set.
5. Manual smoke test: a single `POST /runs/` call against a real GitHub repo with `GITHUB_TOKEN` set completes end-to-end and returns a triage recommendation that cites at least one retrieved runbook chunk.
6. Ingestion script runs to completion: `python -m platform.retrieval.ingest --source docs/runbooks/ --db runs.db` exits 0 and populates `runbook_chunks`.
7. No changes to `platform/core/interfaces/llm.py`, `platform/core/interfaces/tool.py`, `platform/core/interfaces/observer.py`, or any V1 pattern executor.

---

## What V2 Does Not Decide

These questions are intentionally deferred to V3:

- How dynamic workflow generation will represent partially-composed workflows in transit.
- Whether the retriever should be exposed as a first-class tool (callable by agents) vs. always an orchestrator-level pre-processing step.
- Whether SQLite is the right long-term store or whether the `IRunStore` / `IMemoryStore` interfaces are sufficient to swap in PostgreSQL with no further architectural changes.
- Real-time run progress visibility (SSE / WebSocket).

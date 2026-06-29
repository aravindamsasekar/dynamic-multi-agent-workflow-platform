# Platform Architecture

## Overview

The Dynamic Multi-Agent Workflow Platform is a Python service that runs multi-agent AI workflows. Workflows are defined entirely in YAML — no platform code changes are needed to add a new workflow.

## Layers

```
api/              FastAPI — HTTP routing, dependency injection, request/response schemas
platform/
  orchestrator/   Orchestrator + RunManager — workflow lifecycle, state transitions
  patterns/       PatternExecutors — parallel_specialist, router, planner_executor_observer
  agent/          AgentRuntime — drives a single agent through the LLM + tool loop
  config/         ConfigLoader + ConfigValidator — YAML parsing, adapter wiring
  core/           Interfaces, models, exceptions (no business logic)
  knowledge/      RAG pipeline — Document → Chunk → Embed → FAISS → Retrieve
  llm/            ILLMProvider implementations (OpenAI, Mock)
  tools/          IToolAdapter implementations (HTTP, GitHub, MCP, Knowledge, Mock)
  persistence/    SQLAlchemy models, repositories, DB wiring
  registries/     WorkflowRegistry, AgentRegistry, ToolRegistry (in-memory, startup-loaded)
  policy/         PolicyEngine + rules (pre/post-agent hooks)
  observability/  IObserver — ConsoleObserver, PersistingObserver, CompositeObserver
  memory/         IMemoryStore — InMemoryStore (per-run cross-agent key-value)
  state/          SharedState — cross-agent persistent state within a run
  hitl/           ApprovalManager — pause/resume for human-in-the-loop
workflows/        YAML workflow definitions (one directory per workflow)
resources/        Static knowledge documents, prompt templates (not generated)
data/             Generated artifacts — FAISS indexes, manifests (gitignored)
```

## Key Invariants

### AgentRuntime is not modified by pattern executors

`AgentRuntime` drives the LLM+tool loop for a single agent. Pattern executors orchestrate *multiple* agent runtimes but never subclass or monkey-patch the runtime.

### Registries are read-only at runtime

`WorkflowRegistry`, `AgentRegistry`, and `ToolRegistry` are populated once at startup by `ConfigLoader`. They are never written to during request handling.

### ConfigLoader is the single wiring point

Every `IToolAdapter` is constructed by `ConfigLoader._ADAPTER_BUILDERS` (or `_build_knowledge_adapter` for the knowledge type). No adapter is instantiated outside this path.

### Knowledge layer never touches AgentRuntime

The knowledge layer (`platform/knowledge/`) is exposed exclusively as a normal `IToolAdapter` (`KnowledgeAdapter`). `AgentRuntime` calls it like any other tool, with no special casing.

## Data Flow — PR Review with RAG

```
POST /runs/ {workflow_id: "pr_review", input: {...}}
  → Orchestrator serialises input → selects ParallelSpecialistExecutor
  → github_fetch_agent: LLM calls github_get_pr / github_get_files / github_get_diff
      → GitHubAdapter → GitHub REST API
      → returns structured PR summary
  → review_agent: receives PR summary as context
      → calls knowledge_search → KnowledgeAdapter → KnowledgeService
          → KnowledgeRetriever.retrieve(query, collections, top_k)
              → OpenAIEmbedder.embed([query]) → OpenAI Embeddings API
              → FAISSVectorStore.query() → FAISS IndexFlatIP (inner product / cosine sim)
          → returns top-k SearchResult objects
      → formats results + PR data into final structured review
  → RunManager marks run COMPLETED
```

## Knowledge Layer Architecture

```
KnowledgeConfig       — parsed from knowledge_config.yaml
DocumentLoader        — reads files from resources/knowledge/**/
TextChunker           — sliding window, preserves paragraph boundaries
OpenAIEmbedder        — text-embedding-3-small, L2-normalised vectors
FAISSVectorStore      — IndexIDMap2(IndexFlatIP), per-collection .index files
KnowledgeRepository   — SQLite metadata (collection, source_file, chunk_index, hash)
ChunkStore            — per-collection JSON {id: {text, source_file}}
KnowledgeIndexer      — manifest-based incremental indexing (rebuild on any file change)
KnowledgeRetriever    — embeds query, fans out to N collections, merges by score
KnowledgeService      — thin seam over retriever (future: reranking, caching)
KnowledgeAdapter      — IToolAdapter, formats SearchResult → ToolResult.content
```

## Extension Points

| What to extend | How |
|---|---|
| New workflow pattern | Implement `IPatternExecutor`, register in `PatternRegistry` |
| New LLM provider | Implement `ILLMProvider` in `platform/llm/` |
| New tool type | Implement `IToolAdapter`, add `AdapterType` enum value, register builder in `ConfigLoader` |
| New knowledge source | Add document parser in `DocumentLoader.load_file()` |
| New retrieval strategy | Extend `KnowledgeService` (reranking, hybrid search) without touching `KnowledgeRetriever` |

# PR Review Workflow Runbook

## Overview

The `pr_review` workflow uses a two-agent parallel specialist pattern:

1. **github_fetch_agent** — Fetches PR metadata, changed files, and unified diff from GitHub.
2. **review_agent** — Receives the fetched data, searches the knowledge base for relevant coding standards, and generates a structured review.

## Running a PR Review

### Via API

```bash
curl -s -X POST http://localhost:8000/runs/ \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_id": "pr_review",
    "input": {
      "owner": "octocat",
      "repo": "Hello-World",
      "pull_number": 42
    }
  }' | python -m json.tool
```

### Poll for result

```bash
curl http://localhost:8000/runs/{run_id}
```

## Prerequisites

### OPENAI_API_KEY

Required for the review_agent (LLM calls) and for the knowledge embedding layer.

```
OPENAI_API_KEY=sk-...
```

### GITHUB_TOKEN (optional for public repos)

Required for private repositories. Create a fine-grained PAT with read access to:
- Repository contents
- Pull requests

```
GITHUB_TOKEN=github_pat_...
```

### Knowledge Base Indexed

The knowledge base must be indexed before the review_agent can retrieve coding standards.

Check if the index exists:
```bash
ls data/knowledge/
# Expected: coding-standards.index, architecture.index, coding-standards.chunks.json, etc.
```

Run the indexing script if files are missing or knowledge docs have changed:
```bash
python -m scripts.index_knowledge
```

Indexing runs automatically at server startup. The indexer uses SHA-256 manifests to skip unchanged collections — only changed files trigger a rebuild.

## Troubleshooting

### "Knowledge service not configured" (503)

`knowledge_config.yaml` is missing from the project root, or the file failed to parse. Check:
```bash
cat knowledge_config.yaml
python -c "import yaml; print(yaml.safe_load(open('knowledge_config.yaml').read()))"
```

### Review agent returns generic advice (no standards cited)

The knowledge base may be empty or stale. Re-run:
```bash
python -m scripts.index_knowledge
```

If collections show "up to date" but content seems wrong, force a rebuild by deleting the manifest files:
```bash
rm -rf data/knowledge/manifests/
python -m scripts.index_knowledge
```

### GitHub 401 errors

`GITHUB_TOKEN` is missing or expired for a private repo. Generate a new token.

### GitHub 404 errors

The owner/repo/pull_number combination does not exist or you do not have access.

## Checking the Knowledge Search API

Test that the knowledge base is indexed and searchable:

```bash
curl -s -X POST http://localhost:8000/knowledge/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "missing tests in pull request",
    "collections": ["coding-standards"],
    "top_k": 3
  }' | python -m json.tool
```

Expected response:
```json
{
  "query": "missing tests in pull request",
  "results": [
    {
      "score": 0.87,
      "collection": "coding-standards",
      "source_file": "resources/knowledge/coding-standards/pr_review_guidelines.md",
      "chunk_index": 1,
      "text": "Every non-trivial change to business logic must include tests..."
    }
  ]
}
```

## Knowledge Base Management

### Adding new documents

1. Place `.md`, `.txt`, `.py`, or other supported files under `resources/knowledge/<collection>/`
2. Run `python -m scripts.index_knowledge` (or restart the server)

### Viewing indexed collections

```bash
curl http://localhost:8000/knowledge/collections
```

### Viewing a specific collection

```bash
curl http://localhost:8000/knowledge/collections/coding-standards
```

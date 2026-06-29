#!/usr/bin/env python3
"""CLI script to build or rebuild knowledge indexes.

Reads knowledge_config.yaml from the project root, connects to the same
database as the API server, and runs the same KnowledgeIndexer used at
startup. Unchanged collections (manifest matches) are skipped automatically.

Usage:
    python -m scripts.index_knowledge
    python scripts/index_knowledge.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import yaml

from platform.knowledge.config import KnowledgeConfig
from platform.knowledge.embedder import OpenAIEmbedder
from platform.knowledge.indexer import KnowledgeIndexer
from platform.knowledge.vector_store import FAISSVectorStore
from platform.persistence.database import Base, build_engine, build_session_factory


async def _run(indexer: KnowledgeIndexer) -> int:
    """Core async logic — separated for testability."""
    results = await indexer.index_all()
    if not results:
        print("No collections configured.")
        return 0
    for name, count in results.items():
        if count == 0:
            print(f"  {name}: up to date (skipped)")
        else:
            print(f"  {name}: {count} chunk(s) indexed")
    total = sum(results.values())
    changed = sum(1 for c in results.values() if c > 0)
    print(f"\nDone. {changed}/{len(results)} collection(s) rebuilt, {total} chunk(s) total.")
    return 0


def main() -> None:
    config_path = Path("knowledge_config.yaml")
    if not config_path.exists():
        print(f"[ERROR] {config_path} not found. Create it before indexing.", file=sys.stderr)
        sys.exit(1)

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = KnowledgeConfig.from_dict(data)
    except Exception as exc:
        print(f"[ERROR] Failed to parse {config_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    database_url = os.environ.get("DATABASE_URL", "sqlite:///./workflow.db")
    engine = build_engine(database_url)
    Base.metadata.create_all(engine)
    session_factory = build_session_factory(engine)

    try:
        embedder = OpenAIEmbedder(model=config.embedding_model)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    vs = FAISSVectorStore(Path(config.vector_store_path))
    indexer = KnowledgeIndexer(config, embedder, vs, session_factory)

    sys.exit(asyncio.run(_run(indexer)))


if __name__ == "__main__":
    main()

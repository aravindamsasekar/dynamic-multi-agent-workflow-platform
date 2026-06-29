"""KnowledgeService — thin orchestration layer over KnowledgeRetriever.

Exists as the canonical seam for future cross-cutting concerns:
  - query rewriting / expansion
  - hybrid (keyword + dense) search
  - result reranking
  - caching
  - metadata filtering

In V2, each method delegates directly to the underlying retriever with no
additional logic. Adding any of the above features requires changes only here.
"""

from __future__ import annotations

from platform.knowledge.retriever import KnowledgeRetriever
from platform.knowledge.vector_store import SearchResult


class KnowledgeService:
    """High-level search API consumed by adapters and APIs."""

    def __init__(self, retriever: KnowledgeRetriever) -> None:
        self._retriever = retriever

    async def search(
        self,
        query: str,
        collections: list[str],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Search across the given collections and return ranked results."""
        return await self._retriever.retrieve(query, collections, top_k)

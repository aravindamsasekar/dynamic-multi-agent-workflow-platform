"""KnowledgeRetriever — embed a query and search across one or more collections."""

from __future__ import annotations

from platform.knowledge.embedder import IEmbedder
from platform.knowledge.vector_store import IVectorStore, SearchResult


class KnowledgeRetriever:
    """Converts a text query into a ranked list of SearchResults.

    Embeds the query once, fans out to every requested collection, merges
    all results, and returns the global top-k sorted by score descending.
    """

    def __init__(self, embedder: IEmbedder, vector_store: IVectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    async def retrieve(
        self,
        query: str,
        collections: list[str],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Return up to top_k results across all requested collections.

        Returns an empty list for an empty query or an empty collections list.
        """
        if not query or not collections:
            return []

        vectors = await self._embedder.embed([query])
        query_vector = vectors[0]

        all_results: list[SearchResult] = []
        for collection in collections:
            results = self._vector_store.query(collection, query_vector, top_k)
            for r in results:
                r.collection = collection
            all_results.extend(results)

        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:top_k]

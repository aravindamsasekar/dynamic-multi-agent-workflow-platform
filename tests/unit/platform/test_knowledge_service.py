"""Unit tests for KnowledgeService."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from platform.knowledge.retriever import KnowledgeRetriever
from platform.knowledge.service import KnowledgeService
from platform.knowledge.vector_store import SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(faiss_id: int, score: float) -> SearchResult:
    return SearchResult(faiss_id=faiss_id, text="text", source_file="f.md", score=score)


def _mock_retriever(return_value: list[SearchResult] | None = None) -> MagicMock:
    retriever = MagicMock(spec=KnowledgeRetriever)
    retriever.retrieve = AsyncMock(return_value=return_value or [])
    return retriever


# ---------------------------------------------------------------------------
# Delegation
# ---------------------------------------------------------------------------


class TestKnowledgeServiceDelegation:
    async def test_search_delegates_to_retriever(self):
        retriever = _mock_retriever([_result(1, 0.9)])
        service = KnowledgeService(retriever)
        await service.search("query", ["col-a"], top_k=5)
        retriever.retrieve.assert_called_once_with("query", ["col-a"], 5)

    async def test_search_passes_query_correctly(self):
        retriever = _mock_retriever()
        service = KnowledgeService(retriever)
        await service.search("specific question text", ["col"], top_k=3)
        args = retriever.retrieve.call_args[0]
        assert args[0] == "specific question text"

    async def test_search_passes_collections_correctly(self):
        retriever = _mock_retriever()
        service = KnowledgeService(retriever)
        await service.search("q", ["col-a", "col-b", "col-c"], top_k=5)
        args = retriever.retrieve.call_args[0]
        assert args[1] == ["col-a", "col-b", "col-c"]

    async def test_search_passes_top_k_correctly(self):
        retriever = _mock_retriever()
        service = KnowledgeService(retriever)
        await service.search("q", ["col"], top_k=10)
        args = retriever.retrieve.call_args[0]
        assert args[2] == 10

    async def test_search_returns_retriever_results_unchanged(self):
        expected = [_result(1, 0.9), _result(2, 0.7)]
        retriever = _mock_retriever(expected)
        service = KnowledgeService(retriever)
        results = await service.search("q", ["col"], top_k=5)
        assert results == expected

    async def test_search_returns_empty_when_retriever_returns_empty(self):
        retriever = _mock_retriever([])
        service = KnowledgeService(retriever)
        results = await service.search("q", ["col"], top_k=5)
        assert results == []

    async def test_search_called_once_per_invocation(self):
        retriever = _mock_retriever()
        service = KnowledgeService(retriever)
        await service.search("q", ["col"], top_k=5)
        retriever.retrieve.assert_called_once()

    async def test_multiple_searches_each_delegate(self):
        retriever = _mock_retriever()
        service = KnowledgeService(retriever)
        await service.search("first", ["col"], top_k=5)
        await service.search("second", ["col"], top_k=3)
        assert retriever.retrieve.call_count == 2

    async def test_default_top_k_is_five(self):
        retriever = _mock_retriever()
        service = KnowledgeService(retriever)
        await service.search("q", ["col"])
        args = retriever.retrieve.call_args[0]
        assert args[2] == 5


# ---------------------------------------------------------------------------
# Empty / edge inputs forwarded unchanged
# ---------------------------------------------------------------------------


class TestKnowledgeServiceEdgeCases:
    async def test_empty_query_forwarded_to_retriever(self):
        retriever = _mock_retriever([])
        service = KnowledgeService(retriever)
        result = await service.search("", ["col"], top_k=5)
        retriever.retrieve.assert_called_once_with("", ["col"], 5)
        assert result == []

    async def test_empty_collections_forwarded_to_retriever(self):
        retriever = _mock_retriever([])
        service = KnowledgeService(retriever)
        result = await service.search("query", [], top_k=5)
        retriever.retrieve.assert_called_once_with("query", [], 5)
        assert result == []

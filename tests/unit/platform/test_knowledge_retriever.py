"""Unit tests for KnowledgeRetriever."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from platform.knowledge.embedder import MockEmbedder
from platform.knowledge.retriever import KnowledgeRetriever
from platform.knowledge.vector_store import FAISSVectorStore, IVectorStore, SearchResult

DIMS = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vec(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in values))
    return [x / norm for x in values]


def _mock_store(*result_lists: list[SearchResult]) -> MagicMock:
    """Build a mock IVectorStore whose query() returns successive result lists."""
    store = MagicMock(spec=IVectorStore)
    store.query.side_effect = list(result_lists)
    return store


def _result(faiss_id: int, score: float, text: str = "t", src: str = "f.md") -> SearchResult:
    return SearchResult(faiss_id=faiss_id, text=text, source_file=src, score=score)


# ---------------------------------------------------------------------------
# Empty / degenerate inputs
# ---------------------------------------------------------------------------


class TestRetrieverEdgeCases:
    async def test_empty_query_returns_empty(self, tmp_path: Path):
        store = _mock_store()
        embedder = MockEmbedder(dimensions=DIMS)
        retriever = KnowledgeRetriever(embedder=embedder, vector_store=store)
        result = await retriever.retrieve("", ["col-a"], top_k=5)
        assert result == []
        store.query.assert_not_called()

    async def test_empty_collections_returns_empty(self, tmp_path: Path):
        store = _mock_store()
        embedder = MockEmbedder(dimensions=DIMS)
        retriever = KnowledgeRetriever(embedder=embedder, vector_store=store)
        result = await retriever.retrieve("hello", [], top_k=5)
        assert result == []
        store.query.assert_not_called()

    async def test_vector_store_returns_empty_gives_empty_result(self):
        store = _mock_store([])
        embedder = MockEmbedder(dimensions=DIMS)
        retriever = KnowledgeRetriever(embedder=embedder, vector_store=store)
        result = await retriever.retrieve("query", ["col-a"], top_k=5)
        assert result == []


# ---------------------------------------------------------------------------
# Single collection
# ---------------------------------------------------------------------------


class TestRetrieverSingleCollection:
    async def test_returns_results_from_single_collection(self):
        expected = [_result(1, 0.9), _result(2, 0.7)]
        store = _mock_store(expected)
        retriever = KnowledgeRetriever(MockEmbedder(DIMS), store)
        results = await retriever.retrieve("query", ["col-a"], top_k=5)
        assert len(results) == 2
        assert results[0].faiss_id == 1
        assert results[1].faiss_id == 2

    async def test_returns_search_result_objects(self):
        store = _mock_store([_result(1, 0.8)])
        retriever = KnowledgeRetriever(MockEmbedder(DIMS), store)
        results = await retriever.retrieve("query", ["col-a"], top_k=5)
        assert all(isinstance(r, SearchResult) for r in results)

    async def test_vector_store_queried_with_correct_collection(self):
        store = _mock_store([])
        retriever = KnowledgeRetriever(MockEmbedder(DIMS), store)
        await retriever.retrieve("hello world", ["my-collection"], top_k=3)
        assert store.query.call_count == 1
        args = store.query.call_args
        assert args[0][0] == "my-collection"
        assert args[0][2] == 3  # top_k passed through

    async def test_vector_store_queried_with_embedded_vector(self):
        embedder = MockEmbedder(dimensions=DIMS)
        store = _mock_store([])
        retriever = KnowledgeRetriever(embedder, store)
        await retriever.retrieve("specific text", ["col"], top_k=5)

        expected_vec = (await embedder.embed(["specific text"]))[0]
        actual_vec = list(store.query.call_args[0][1])
        assert actual_vec == pytest.approx(expected_vec, abs=1e-5)


# ---------------------------------------------------------------------------
# Multi-collection
# ---------------------------------------------------------------------------


class TestRetrieverMultiCollection:
    async def test_queries_all_collections(self):
        store = _mock_store([], [], [])
        retriever = KnowledgeRetriever(MockEmbedder(DIMS), store)
        await retriever.retrieve("query", ["col-a", "col-b", "col-c"], top_k=5)
        assert store.query.call_count == 3

    async def test_queries_collections_in_order(self):
        store = _mock_store([], [], [])
        retriever = KnowledgeRetriever(MockEmbedder(DIMS), store)
        await retriever.retrieve("query", ["first", "second", "third"], top_k=5)
        called_collections = [c[0][0] for c in store.query.call_args_list]
        assert called_collections == ["first", "second", "third"]

    async def test_embed_called_once_for_multi_collection(self):
        embedder = MockEmbedder(DIMS)
        embedder.embed = AsyncMock(return_value=[[0.5, 0.5, 0.5, 0.5]])
        store = _mock_store([], [])
        retriever = KnowledgeRetriever(embedder, store)
        await retriever.retrieve("query", ["col-a", "col-b"], top_k=5)
        embedder.embed.assert_called_once_with(["query"])

    async def test_results_merged_from_multiple_collections(self):
        col_a_results = [_result(1, 0.9, text="from-a")]
        col_b_results = [_result(2, 0.7, text="from-b")]
        store = _mock_store(col_a_results, col_b_results)
        retriever = KnowledgeRetriever(MockEmbedder(DIMS), store)
        results = await retriever.retrieve("query", ["col-a", "col-b"], top_k=10)
        texts = {r.text for r in results}
        assert "from-a" in texts
        assert "from-b" in texts

    async def test_same_query_vector_sent_to_all_collections(self):
        embedder = MockEmbedder(DIMS)
        fixed_vec = [0.1, 0.2, 0.3, 0.4]
        embedder.embed = AsyncMock(return_value=[fixed_vec])
        store = _mock_store([], [])
        retriever = KnowledgeRetriever(embedder, store)
        await retriever.retrieve("query", ["col-a", "col-b"], top_k=5)

        for c in store.query.call_args_list:
            assert list(c[0][1]) == fixed_vec


# ---------------------------------------------------------------------------
# Score sorting and top_k
# ---------------------------------------------------------------------------


class TestRetrieverSortingAndTopK:
    async def test_results_sorted_by_score_descending(self):
        results = [
            _result(3, 0.5),
            _result(1, 0.9),
            _result(2, 0.7),
        ]
        store = _mock_store(results)
        retriever = KnowledgeRetriever(MockEmbedder(DIMS), store)
        output = await retriever.retrieve("query", ["col"], top_k=10)
        scores = [r.score for r in output]
        assert scores == sorted(scores, reverse=True)

    async def test_top_k_limits_results(self):
        results = [_result(i, float(i) / 10) for i in range(1, 9)]
        store = _mock_store(results)
        retriever = KnowledgeRetriever(MockEmbedder(DIMS), store)
        output = await retriever.retrieve("query", ["col"], top_k=3)
        assert len(output) == 3

    async def test_top_k_returns_highest_scores(self):
        results = [_result(i, float(i) / 10) for i in range(1, 9)]
        store = _mock_store(results)
        retriever = KnowledgeRetriever(MockEmbedder(DIMS), store)
        output = await retriever.retrieve("query", ["col"], top_k=3)
        min_output_score = min(r.score for r in output)
        assert min_output_score >= 0.6  # top 3 of 0.1..0.8 are 0.6, 0.7, 0.8

    async def test_fewer_results_than_top_k_returns_all(self):
        results = [_result(1, 0.9), _result(2, 0.8)]
        store = _mock_store(results)
        retriever = KnowledgeRetriever(MockEmbedder(DIMS), store)
        output = await retriever.retrieve("query", ["col"], top_k=10)
        assert len(output) == 2

    async def test_multi_collection_top_k_selects_globally_best(self):
        col_a = [_result(1, 0.9, text="a-top"), _result(2, 0.3, text="a-low")]
        col_b = [_result(3, 0.8, text="b-top"), _result(4, 0.2, text="b-low")]
        store = _mock_store(col_a, col_b)
        retriever = KnowledgeRetriever(MockEmbedder(DIMS), store)
        output = await retriever.retrieve("query", ["col-a", "col-b"], top_k=2)
        texts = {r.text for r in output}
        assert texts == {"a-top", "b-top"}

    async def test_top_k_one_returns_single_best(self):
        results = [_result(1, 0.9), _result(2, 0.5), _result(3, 0.1)]
        store = _mock_store(results)
        retriever = KnowledgeRetriever(MockEmbedder(DIMS), store)
        output = await retriever.retrieve("query", ["col"], top_k=1)
        assert len(output) == 1
        assert output[0].score == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Integration: real FAISS + MockEmbedder
# ---------------------------------------------------------------------------


class TestRetrieverWithRealComponents:
    """End-to-end pipeline: MockEmbedder + FAISSVectorStore + KnowledgeRetriever."""

    async def test_retrieves_inserted_chunk(self, tmp_path: Path):
        embedder = MockEmbedder(dimensions=DIMS)
        vs = FAISSVectorStore(tmp_path, dimensions=DIMS)

        text = "The quick brown fox"
        vec = (await embedder.embed([text]))[0]
        vs.upsert("docs", [1], [vec], [text], ["fox.md"])

        retriever = KnowledgeRetriever(embedder, vs)
        results = await retriever.retrieve(text, ["docs"], top_k=1)

        assert len(results) == 1
        assert results[0].text == text
        assert results[0].score == pytest.approx(1.0, abs=1e-4)

    async def test_multi_collection_integration(self, tmp_path: Path):
        embedder = MockEmbedder(dimensions=DIMS)
        vs = FAISSVectorStore(tmp_path, dimensions=DIMS)

        for col, txt in [("col-a", "alpha document"), ("col-b", "beta document")]:
            vec = (await embedder.embed([txt]))[0]
            vs.upsert(col, [hash(txt) % 10000 + 1], [vec], [txt], ["src.md"])

        retriever = KnowledgeRetriever(embedder, vs)
        results = await retriever.retrieve("alpha document", ["col-a", "col-b"], top_k=5)

        assert len(results) >= 1
        assert results[0].text == "alpha document"

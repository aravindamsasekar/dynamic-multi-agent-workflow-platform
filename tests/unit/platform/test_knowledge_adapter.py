"""Unit tests for KnowledgeAdapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from platform.core.models.tool import ToolCall
from platform.knowledge.service import KnowledgeService
from platform.knowledge.vector_store import SearchResult
from platform.tools.knowledge_adapter import KnowledgeAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _call(query: str | None = "test query", tool_id: str = "tool-1") -> ToolCall:
    inp: dict = {}
    if query is not None:
        inp["query"] = query
    return ToolCall(tool_use_id=tool_id, tool_name="knowledge_search", input=inp)


def _result(
    faiss_id: int = 1,
    text: str = "chunk text",
    source_file: str = "docs/guide.md",
    score: float = 0.9,
    collection: str = "col-a",
) -> SearchResult:
    return SearchResult(
        faiss_id=faiss_id,
        text=text,
        source_file=source_file,
        score=score,
        collection=collection,
    )


def _mock_service(results: list[SearchResult] | None = None) -> MagicMock:
    service = MagicMock(spec=KnowledgeService)
    service.search = AsyncMock(return_value=results or [])
    return service


# ---------------------------------------------------------------------------
# Missing query
# ---------------------------------------------------------------------------


class TestKnowledgeAdapterMissingQuery:
    async def test_missing_query_key_returns_error(self):
        call = ToolCall(tool_use_id="t1", tool_name="ks", input={})
        adapter = KnowledgeAdapter(_mock_service(), ["col-a"])
        result = await adapter.execute(call)
        assert result.is_error is True
        assert "query" in result.content.lower()

    async def test_empty_string_query_returns_error(self):
        adapter = KnowledgeAdapter(_mock_service(), ["col-a"])
        result = await adapter.execute(_call(query=""))
        assert result.is_error is True

    async def test_missing_query_does_not_call_service(self):
        service = _mock_service()
        adapter = KnowledgeAdapter(service, ["col-a"])
        await adapter.execute(ToolCall(tool_use_id="t", tool_name="ks", input={}))
        service.search.assert_not_called()

    async def test_missing_query_uses_correct_tool_use_id(self):
        adapter = KnowledgeAdapter(_mock_service(), ["col-a"])
        result = await adapter.execute(
            ToolCall(tool_use_id="my-id", tool_name="ks", input={})
        )
        assert result.tool_use_id == "my-id"


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


class TestKnowledgeAdapterSuccess:
    async def test_returns_non_error_result(self):
        service = _mock_service([_result()])
        adapter = KnowledgeAdapter(service, ["col-a"])
        result = await adapter.execute(_call())
        assert result.is_error is False

    async def test_calls_service_with_query_and_collections(self):
        service = _mock_service([_result()])
        adapter = KnowledgeAdapter(service, ["col-a", "col-b"], top_k=3)
        await adapter.execute(_call("my query"))
        service.search.assert_called_once_with("my query", ["col-a", "col-b"], 3)

    async def test_tool_use_id_preserved(self):
        service = _mock_service([_result()])
        adapter = KnowledgeAdapter(service, ["col-a"])
        result = await adapter.execute(_call(tool_id="tid-42"))
        assert result.tool_use_id == "tid-42"

    async def test_content_includes_chunk_text(self):
        service = _mock_service([_result(text="The quick brown fox")])
        adapter = KnowledgeAdapter(service, ["col-a"])
        result = await adapter.execute(_call())
        assert "The quick brown fox" in result.content

    async def test_content_includes_source_file(self):
        service = _mock_service([_result(source_file="guides/setup.md")])
        adapter = KnowledgeAdapter(service, ["col-a"])
        result = await adapter.execute(_call())
        assert "guides/setup.md" in result.content

    async def test_content_includes_collection(self):
        service = _mock_service([_result(collection="architecture")])
        adapter = KnowledgeAdapter(service, ["architecture"])
        result = await adapter.execute(_call())
        assert "architecture" in result.content

    async def test_content_includes_score(self):
        service = _mock_service([_result(score=0.8765)])
        adapter = KnowledgeAdapter(service, ["col-a"])
        result = await adapter.execute(_call())
        assert "0.8765" in result.content

    async def test_multiple_results_all_included(self):
        results = [
            _result(faiss_id=1, text="first chunk", score=0.9),
            _result(faiss_id=2, text="second chunk", score=0.7),
        ]
        service = _mock_service(results)
        adapter = KnowledgeAdapter(service, ["col-a"])
        result = await adapter.execute(_call())
        assert "first chunk" in result.content
        assert "second chunk" in result.content

    async def test_content_includes_result_count(self):
        service = _mock_service([_result(), _result(faiss_id=2), _result(faiss_id=3)])
        adapter = KnowledgeAdapter(service, ["col-a"])
        result = await adapter.execute(_call())
        assert "3" in result.content

    async def test_default_top_k_is_five(self):
        service = _mock_service([])
        adapter = KnowledgeAdapter(service, ["col-a"])
        await adapter.execute(_call("q"))
        service.search.assert_called_once_with("q", ["col-a"], 5)


# ---------------------------------------------------------------------------
# Empty results
# ---------------------------------------------------------------------------


class TestKnowledgeAdapterEmptyResults:
    async def test_empty_results_returns_non_error(self):
        service = _mock_service([])
        adapter = KnowledgeAdapter(service, ["col-a"])
        result = await adapter.execute(_call())
        assert result.is_error is False

    async def test_empty_results_content_indicates_no_results(self):
        service = _mock_service([])
        adapter = KnowledgeAdapter(service, ["col-a"])
        result = await adapter.execute(_call("unknown topic"))
        assert "no results" in result.content.lower() or "not found" in result.content.lower()

    async def test_empty_results_tool_use_id_preserved(self):
        service = _mock_service([])
        adapter = KnowledgeAdapter(service, ["col-a"])
        result = await adapter.execute(_call(tool_id="empty-id"))
        assert result.tool_use_id == "empty-id"


# ---------------------------------------------------------------------------
# Service error handling
# ---------------------------------------------------------------------------


class TestKnowledgeAdapterServiceError:
    async def test_service_exception_returns_error(self):
        service = MagicMock(spec=KnowledgeService)
        service.search = AsyncMock(side_effect=RuntimeError("FAISS index corrupt"))
        adapter = KnowledgeAdapter(service, ["col-a"])
        result = await adapter.execute(_call())
        assert result.is_error is True

    async def test_service_exception_message_in_content(self):
        service = MagicMock(spec=KnowledgeService)
        service.search = AsyncMock(side_effect=RuntimeError("FAISS index corrupt"))
        adapter = KnowledgeAdapter(service, ["col-a"])
        result = await adapter.execute(_call())
        assert "FAISS index corrupt" in result.content

    async def test_service_exception_uses_correct_tool_use_id(self):
        service = MagicMock(spec=KnowledgeService)
        service.search = AsyncMock(side_effect=ValueError("bad config"))
        adapter = KnowledgeAdapter(service, ["col-a"])
        result = await adapter.execute(_call(tool_id="err-id"))
        assert result.tool_use_id == "err-id"

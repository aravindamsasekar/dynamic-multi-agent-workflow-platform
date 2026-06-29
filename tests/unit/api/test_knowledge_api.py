"""Unit tests for knowledge API endpoints.

Uses a test-only FastAPI app with dependency overrides; no network calls,
no real FAISS, no OpenAI.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.dependencies import get_db_session, get_knowledge_service
from api.routers import knowledge as knowledge_router
from platform.knowledge.service import KnowledgeService
from platform.knowledge.vector_store import SearchResult
from platform.persistence.database import Base
from platform.persistence.repositories.knowledge_repo import KnowledgeRepository

# ---------------------------------------------------------------------------
# In-memory DB helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def db_factory():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


@pytest.fixture
def seeded_db(db_factory):
    """Seed SQLite with two collections worth of knowledge_chunk rows."""
    repo = KnowledgeRepository()
    with db_factory() as s:
        # col-a: 2 docs, 3 chunks
        id1 = repo.insert(s, "col-a", "docs/guide.md", 0, "h1")
        id2 = repo.insert(s, "col-a", "docs/guide.md", 1, "h2")
        id3 = repo.insert(s, "col-a", "docs/api.md", 0, "h3")
        # col-b: 1 doc, 2 chunks
        id4 = repo.insert(s, "col-b", "src/main.py", 0, "h4")
        id5 = repo.insert(s, "col-b", "src/main.py", 1, "h5")
        s.commit()
    return db_factory, {"id1": id1, "id2": id2, "id3": id3, "id4": id4, "id5": id5}


# ---------------------------------------------------------------------------
# Mock KnowledgeService
# ---------------------------------------------------------------------------


def _mock_ks(results: list[SearchResult] | None = None) -> MagicMock:
    ks = MagicMock(spec=KnowledgeService)
    ks.search = AsyncMock(return_value=results or [])
    return ks


# ---------------------------------------------------------------------------
# Test FastAPI app
# ---------------------------------------------------------------------------


_test_app = FastAPI()
_test_app.include_router(knowledge_router.router, prefix="/knowledge", tags=["knowledge"])


def _db_override(factory):
    def _dep():
        with factory() as s:
            yield s
    return _dep


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def client_with_service(seeded_db):
    db_factory, ids = seeded_db
    ks = _mock_ks()
    _test_app.dependency_overrides[get_db_session] = _db_override(db_factory)
    _test_app.dependency_overrides[get_knowledge_service] = lambda: ks
    async with AsyncClient(
        transport=ASGITransport(app=_test_app), base_url="http://test"
    ) as c:
        yield c, ks, ids
    _test_app.dependency_overrides.clear()


@pytest.fixture
async def client_no_service(seeded_db):
    db_factory, ids = seeded_db
    _test_app.dependency_overrides[get_db_session] = _db_override(db_factory)
    _test_app.dependency_overrides[get_knowledge_service] = lambda: None
    async with AsyncClient(
        transport=ASGITransport(app=_test_app), base_url="http://test"
    ) as c:
        yield c
    _test_app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# GET /knowledge/collections
# ---------------------------------------------------------------------------


class TestListCollections:
    async def test_returns_200_with_service(self, client_with_service):
        client, ks, ids = client_with_service
        resp = await client.get("/knowledge/collections")
        assert resp.status_code == 200

    async def test_returns_503_without_service(self, client_no_service):
        resp = await client_no_service.get("/knowledge/collections")
        assert resp.status_code == 503

    async def test_returns_list(self, client_with_service):
        client, ks, ids = client_with_service
        resp = await client.get("/knowledge/collections")
        data = resp.json()
        assert isinstance(data, list)

    async def test_contains_both_collections(self, client_with_service):
        client, ks, ids = client_with_service
        resp = await client.get("/knowledge/collections")
        names = {c["name"] for c in resp.json()}
        assert "col-a" in names
        assert "col-b" in names

    async def test_chunk_count_correct(self, client_with_service):
        client, ks, ids = client_with_service
        resp = await client.get("/knowledge/collections")
        by_name = {c["name"]: c for c in resp.json()}
        assert by_name["col-a"]["chunk_count"] == 3
        assert by_name["col-b"]["chunk_count"] == 2

    async def test_document_count_correct(self, client_with_service):
        client, ks, ids = client_with_service
        resp = await client.get("/knowledge/collections")
        by_name = {c["name"]: c for c in resp.json()}
        assert by_name["col-a"]["document_count"] == 2  # guide.md + api.md
        assert by_name["col-b"]["document_count"] == 1  # main.py only


# ---------------------------------------------------------------------------
# GET /knowledge/collections/{collection}
# ---------------------------------------------------------------------------


class TestGetCollection:
    async def test_returns_200_for_known_collection(self, client_with_service):
        client, ks, ids = client_with_service
        resp = await client.get("/knowledge/collections/col-a")
        assert resp.status_code == 200

    async def test_returns_404_for_unknown_collection(self, client_with_service):
        client, ks, ids = client_with_service
        resp = await client.get("/knowledge/collections/does-not-exist")
        assert resp.status_code == 404

    async def test_returns_503_without_service(self, client_no_service):
        resp = await client_no_service.get("/knowledge/collections/col-a")
        assert resp.status_code == 503

    async def test_response_has_name(self, client_with_service):
        client, ks, ids = client_with_service
        resp = await client.get("/knowledge/collections/col-a")
        assert resp.json()["name"] == "col-a"

    async def test_response_has_chunk_count(self, client_with_service):
        client, ks, ids = client_with_service
        resp = await client.get("/knowledge/collections/col-a")
        assert resp.json()["chunk_count"] == 3

    async def test_response_has_documents_list(self, client_with_service):
        client, ks, ids = client_with_service
        resp = await client.get("/knowledge/collections/col-a")
        docs = resp.json()["documents"]
        assert isinstance(docs, list)
        assert "docs/guide.md" in docs
        assert "docs/api.md" in docs

    async def test_col_b_detail(self, client_with_service):
        client, ks, ids = client_with_service
        resp = await client.get("/knowledge/collections/col-b")
        data = resp.json()
        assert data["name"] == "col-b"
        assert data["chunk_count"] == 2
        assert data["documents"] == ["src/main.py"]


# ---------------------------------------------------------------------------
# POST /knowledge/search
# ---------------------------------------------------------------------------


class TestSearch:
    async def test_returns_503_without_service(self, client_no_service):
        resp = await client_no_service.post(
            "/knowledge/search",
            json={"query": "hello", "collections": ["col-a"]},
        )
        assert resp.status_code == 503

    async def test_returns_200_with_empty_results(self, client_with_service):
        client, ks, ids = client_with_service
        ks.search.return_value = []
        resp = await client.post(
            "/knowledge/search",
            json={"query": "unknown", "collections": ["col-a"]},
        )
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    async def test_search_calls_service(self, client_with_service):
        client, ks, ids = client_with_service
        ks.search.return_value = []
        await client.post(
            "/knowledge/search",
            json={"query": "my query", "collections": ["col-a", "col-b"], "top_k": 3},
        )
        ks.search.assert_called_once_with("my query", ["col-a", "col-b"], 3)

    async def test_search_returns_query_in_response(self, client_with_service):
        client, ks, ids = client_with_service
        ks.search.return_value = []
        resp = await client.post(
            "/knowledge/search",
            json={"query": "my question", "collections": ["col-a"]},
        )
        assert resp.json()["query"] == "my question"

    async def test_search_returns_results_with_chunk_index(self, client_with_service):
        client, ks, ids = client_with_service
        # Use real IDs that exist in the seeded DB
        sr = SearchResult(
            faiss_id=ids["id1"],
            text="chunk content",
            source_file="docs/guide.md",
            score=0.95,
            collection="col-a",
        )
        ks.search.return_value = [sr]
        resp = await client.post(
            "/knowledge/search",
            json={"query": "guide", "collections": ["col-a"]},
        )
        results = resp.json()["results"]
        assert len(results) == 1
        assert results[0]["chunk_index"] == 0  # seeded with chunk_index=0
        assert results[0]["source_file"] == "docs/guide.md"
        assert results[0]["collection"] == "col-a"
        assert results[0]["text"] == "chunk content"
        assert abs(results[0]["score"] - 0.95) < 0.001

    async def test_search_result_chunk_index_for_second_chunk(self, client_with_service):
        client, ks, ids = client_with_service
        sr = SearchResult(
            faiss_id=ids["id2"],
            text="second chunk",
            source_file="docs/guide.md",
            score=0.8,
            collection="col-a",
        )
        ks.search.return_value = [sr]
        resp = await client.post(
            "/knowledge/search",
            json={"query": "guide", "collections": ["col-a"]},
        )
        results = resp.json()["results"]
        assert results[0]["chunk_index"] == 1  # second chunk

    async def test_search_default_top_k_five(self, client_with_service):
        client, ks, ids = client_with_service
        ks.search.return_value = []
        await client.post(
            "/knowledge/search",
            json={"query": "q", "collections": ["col-a"]},
        )
        ks.search.assert_called_once_with("q", ["col-a"], 5)

    async def test_search_top_k_validation_rejects_zero(self, client_with_service):
        client, ks, ids = client_with_service
        resp = await client.post(
            "/knowledge/search",
            json={"query": "q", "collections": ["col-a"], "top_k": 0},
        )
        assert resp.status_code == 422

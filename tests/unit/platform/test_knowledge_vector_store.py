"""Unit tests for FAISSVectorStore, _ChunkStore, and SearchResult."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from platform.knowledge.vector_store import (
    FAISSVectorStore,
    IVectorStore,
    SearchResult,
    _ChunkEntry,
    _ChunkStore,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIMS = 4  # small dimension for fast tests


def unit_vec(values: list[float]) -> list[float]:
    """Return an L2-normalised vector."""
    norm = math.sqrt(sum(x * x for x in values))
    return [x / norm for x in values]


def make_vecs(n: int) -> list[list[float]]:
    """Return n distinct normalised vectors in DIMS dimensions."""
    return [unit_vec([float(i + 1), float(i + 2), float(i + 3), float(i + 4)]) for i in range(n)]


@pytest.fixture
def store(tmp_path: Path) -> FAISSVectorStore:
    return FAISSVectorStore(base_path=tmp_path, dimensions=DIMS)


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_stores_all_fields(self):
        r = SearchResult(faiss_id=1, text="hello", source_file="f.md", score=0.9)
        assert r.faiss_id == 1
        assert r.text == "hello"
        assert r.source_file == "f.md"
        assert r.score == 0.9


# ---------------------------------------------------------------------------
# IVectorStore is abstract
# ---------------------------------------------------------------------------


class TestIVectorStoreAbstract:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            IVectorStore()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# _ChunkEntry
# ---------------------------------------------------------------------------


class TestChunkEntry:
    def test_stores_text_and_source_file(self):
        e = _ChunkEntry(text="content", source_file="docs/guide.md")
        assert e.text == "content"
        assert e.source_file == "docs/guide.md"


# ---------------------------------------------------------------------------
# _ChunkStore
# ---------------------------------------------------------------------------


class TestChunkStore:
    def test_load_missing_file_returns_empty(self, tmp_path: Path):
        store = _ChunkStore(tmp_path / "col.chunks.json")
        assert store.load() == {}

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        store = _ChunkStore(tmp_path / "col.chunks.json")
        data = {
            1: _ChunkEntry(text="first chunk", source_file="a.md"),
            2: _ChunkEntry(text="second chunk", source_file="b.md"),
        }
        store.save(data)
        loaded = store.load()
        assert loaded == data

    def test_save_is_atomic(self, tmp_path: Path):
        """The target file is replaced atomically — no .tmp file left behind."""
        store = _ChunkStore(tmp_path / "col.chunks.json")
        store.save({1: _ChunkEntry(text="x", source_file="f.md")})
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []

    def test_upsert_merges_with_existing(self, tmp_path: Path):
        store = _ChunkStore(tmp_path / "col.chunks.json")
        store.save({1: _ChunkEntry(text="old", source_file="f.md")})
        store.upsert({2: _ChunkEntry(text="new", source_file="g.md")})
        loaded = store.load()
        assert 1 in loaded
        assert 2 in loaded
        assert loaded[1].text == "old"
        assert loaded[2].text == "new"

    def test_upsert_overwrites_existing_id(self, tmp_path: Path):
        store = _ChunkStore(tmp_path / "col.chunks.json")
        store.save({1: _ChunkEntry(text="v1", source_file="f.md")})
        store.upsert({1: _ChunkEntry(text="v2", source_file="f.md")})
        assert store.load()[1].text == "v2"

    def test_remove_ids_deletes_entries(self, tmp_path: Path):
        store = _ChunkStore(tmp_path / "col.chunks.json")
        store.save({
            1: _ChunkEntry(text="a", source_file="f.md"),
            2: _ChunkEntry(text="b", source_file="g.md"),
            3: _ChunkEntry(text="c", source_file="h.md"),
        })
        store.remove_ids([1, 3])
        loaded = store.load()
        assert 1 not in loaded
        assert 3 not in loaded
        assert 2 in loaded

    def test_remove_ids_ignores_missing(self, tmp_path: Path):
        store = _ChunkStore(tmp_path / "col.chunks.json")
        store.save({1: _ChunkEntry(text="x", source_file="f.md")})
        store.remove_ids([999])  # should not raise
        assert 1 in store.load()

    def test_clear_removes_file(self, tmp_path: Path):
        path = tmp_path / "col.chunks.json"
        store = _ChunkStore(path)
        store.save({1: _ChunkEntry(text="x", source_file="f.md")})
        assert path.exists()
        store.clear()
        assert not path.exists()

    def test_clear_on_nonexistent_file_is_safe(self, tmp_path: Path):
        _ChunkStore(tmp_path / "missing.chunks.json").clear()  # should not raise

    def test_corrupt_file_returns_empty(self, tmp_path: Path):
        path = tmp_path / "col.chunks.json"
        path.write_text("not valid json {{{{", encoding="utf-8")
        assert _ChunkStore(path).load() == {}

    def test_keys_survive_json_serialisation(self, tmp_path: Path):
        """JSON keys are strings; load() must convert them back to int."""
        store = _ChunkStore(tmp_path / "col.chunks.json")
        store.save({42: _ChunkEntry(text="text", source_file="f.md")})
        loaded = store.load()
        assert 42 in loaded
        assert "42" not in loaded


# ---------------------------------------------------------------------------
# FAISSVectorStore — construction and file layout
# ---------------------------------------------------------------------------


class TestFAISSVectorStoreConstruction:
    def test_creates_base_directory(self, tmp_path: Path):
        path = tmp_path / "deep" / "dir"
        FAISSVectorStore(base_path=path, dimensions=DIMS)
        assert path.is_dir()

    def test_collection_does_not_exist_initially(self, store: FAISSVectorStore):
        assert store.collection_exists("my-col") is False

    def test_chunk_count_zero_for_nonexistent_collection(self, store: FAISSVectorStore):
        assert store.get_chunk_count("my-col") == 0

    def test_is_ivectorstore_subclass(self, store: FAISSVectorStore):
        assert isinstance(store, IVectorStore)


# ---------------------------------------------------------------------------
# FAISSVectorStore — upsert
# ---------------------------------------------------------------------------


class TestFAISSVectorStoreUpsert:
    def test_upsert_creates_index_file(self, store: FAISSVectorStore, tmp_path: Path):
        vecs = make_vecs(2)
        store.upsert("col", [1, 2], vecs, ["t1", "t2"], ["f1.md", "f2.md"])
        assert (tmp_path / "col.index").exists()

    def test_upsert_creates_chunk_json(self, store: FAISSVectorStore, tmp_path: Path):
        vecs = make_vecs(1)
        store.upsert("col", [1], vecs, ["text"], ["src.md"])
        assert (tmp_path / "col.chunks.json").exists()

    def test_collection_exists_after_upsert(self, store: FAISSVectorStore):
        store.upsert("col", [1], make_vecs(1), ["t"], ["f.md"])
        assert store.collection_exists("col") is True

    def test_chunk_count_after_upsert(self, store: FAISSVectorStore):
        store.upsert("col", [1, 2, 3], make_vecs(3), ["t1", "t2", "t3"], ["f.md"] * 3)
        assert store.get_chunk_count("col") == 3

    def test_empty_upsert_is_noop(self, store: FAISSVectorStore):
        store.upsert("col", [], [], [], [])
        assert store.collection_exists("col") is False

    def test_upsert_accumulates_across_calls(self, store: FAISSVectorStore):
        store.upsert("col", [1], make_vecs(1), ["t1"], ["f.md"])
        store.upsert("col", [2], make_vecs(1)[0:1], ["t2"], ["g.md"])
        assert store.get_chunk_count("col") == 2

    def test_separate_collections_are_independent(self, store: FAISSVectorStore):
        store.upsert("col-a", [1], make_vecs(1), ["t"], ["f.md"])
        store.upsert("col-b", [2, 3], make_vecs(2), ["t1", "t2"], ["f.md", "g.md"])
        assert store.get_chunk_count("col-a") == 1
        assert store.get_chunk_count("col-b") == 2


# ---------------------------------------------------------------------------
# FAISSVectorStore — query
# ---------------------------------------------------------------------------


class TestFAISSVectorStoreQuery:
    def test_query_empty_collection_returns_empty(self, store: FAISSVectorStore):
        results = store.query("nonexistent", make_vecs(1)[0], top_k=5)
        assert results == []

    def test_query_returns_search_result_objects(self, store: FAISSVectorStore):
        vecs = make_vecs(3)
        store.upsert("col", [1, 2, 3], vecs, ["t1", "t2", "t3"], ["f.md", "g.md", "h.md"])
        results = store.query("col", vecs[0], top_k=3)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_query_returns_at_most_top_k(self, store: FAISSVectorStore):
        vecs = make_vecs(5)
        store.upsert("col", list(range(1, 6)), vecs, [f"t{i}" for i in range(5)], ["f.md"] * 5)
        results = store.query("col", vecs[0], top_k=3)
        assert len(results) <= 3

    def test_query_exact_match_has_highest_score(self, store: FAISSVectorStore):
        vecs = make_vecs(3)
        store.upsert("col", [1, 2, 3], vecs, ["t1", "t2", "t3"], ["f.md", "g.md", "h.md"])
        results = store.query("col", vecs[0], top_k=3)
        # Exact match (IP of identical normalised vectors = 1.0)
        top = results[0]
        assert top.faiss_id == 1
        assert abs(top.score - 1.0) < 1e-4

    def test_query_result_contains_correct_text(self, store: FAISSVectorStore):
        vecs = make_vecs(2)
        store.upsert("col", [10, 20], vecs, ["hello", "world"], ["a.md", "b.md"])
        results = store.query("col", vecs[0], top_k=1)
        assert results[0].text == "hello"
        assert results[0].faiss_id == 10

    def test_query_result_contains_source_file(self, store: FAISSVectorStore):
        vecs = make_vecs(1)
        store.upsert("col", [7], vecs, ["text"], ["docs/guide.md"])
        results = store.query("col", vecs[0], top_k=1)
        assert results[0].source_file == "docs/guide.md"

    def test_query_top_k_greater_than_count_returns_all(self, store: FAISSVectorStore):
        vecs = make_vecs(2)
        store.upsert("col", [1, 2], vecs, ["t1", "t2"], ["f.md", "g.md"])
        results = store.query("col", vecs[0], top_k=100)
        assert len(results) == 2

    def test_index_persists_across_store_instances(self, tmp_path: Path):
        vecs = make_vecs(2)
        s1 = FAISSVectorStore(tmp_path, DIMS)
        s1.upsert("col", [1, 2], vecs, ["t1", "t2"], ["f.md", "g.md"])

        s2 = FAISSVectorStore(tmp_path, DIMS)
        results = s2.query("col", vecs[0], top_k=2)
        assert len(results) == 2
        assert results[0].faiss_id == 1


# ---------------------------------------------------------------------------
# FAISSVectorStore — remove_ids
# ---------------------------------------------------------------------------


class TestFAISSVectorStoreRemoveIds:
    def test_remove_reduces_chunk_count(self, store: FAISSVectorStore):
        vecs = make_vecs(3)
        store.upsert("col", [1, 2, 3], vecs, ["t1", "t2", "t3"], ["f.md"] * 3)
        store.remove_ids("col", [1])
        assert store.get_chunk_count("col") == 2

    def test_removed_id_not_in_query_results(self, store: FAISSVectorStore):
        vecs = make_vecs(3)
        store.upsert("col", [1, 2, 3], vecs, ["t1", "t2", "t3"], ["f.md"] * 3)
        store.remove_ids("col", [1])
        results = store.query("col", vecs[0], top_k=3)
        returned_ids = {r.faiss_id for r in results}
        assert 1 not in returned_ids

    def test_remove_ids_clears_chunk_store_entries(self, store: FAISSVectorStore, tmp_path: Path):
        vecs = make_vecs(2)
        store.upsert("col", [1, 2], vecs, ["t1", "t2"], ["f.md", "g.md"])
        store.remove_ids("col", [1])
        chunk_store = _ChunkStore(tmp_path / "col.chunks.json")
        loaded = chunk_store.load()
        assert 1 not in loaded
        assert 2 in loaded

    def test_remove_nonexistent_collection_is_safe(self, store: FAISSVectorStore):
        store.remove_ids("ghost", [1, 2])  # should not raise

    def test_remove_empty_list_is_noop(self, store: FAISSVectorStore):
        vecs = make_vecs(1)
        store.upsert("col", [1], vecs, ["t"], ["f.md"])
        store.remove_ids("col", [])
        assert store.get_chunk_count("col") == 1


# ---------------------------------------------------------------------------
# FAISSVectorStore — delete_collection
# ---------------------------------------------------------------------------


class TestFAISSVectorStoreDeleteCollection:
    def test_deletes_index_file(self, store: FAISSVectorStore, tmp_path: Path):
        store.upsert("col", [1], make_vecs(1), ["t"], ["f.md"])
        store.delete_collection("col")
        assert not (tmp_path / "col.index").exists()

    def test_deletes_chunk_json(self, store: FAISSVectorStore, tmp_path: Path):
        store.upsert("col", [1], make_vecs(1), ["t"], ["f.md"])
        store.delete_collection("col")
        assert not (tmp_path / "col.chunks.json").exists()

    def test_collection_not_exists_after_delete(self, store: FAISSVectorStore):
        store.upsert("col", [1], make_vecs(1), ["t"], ["f.md"])
        store.delete_collection("col")
        assert store.collection_exists("col") is False

    def test_chunk_count_zero_after_delete(self, store: FAISSVectorStore):
        store.upsert("col", [1, 2], make_vecs(2), ["t1", "t2"], ["f.md"] * 2)
        store.delete_collection("col")
        assert store.get_chunk_count("col") == 0

    def test_delete_nonexistent_collection_is_safe(self, store: FAISSVectorStore):
        store.delete_collection("ghost")  # should not raise

    def test_delete_only_removes_target_collection(self, store: FAISSVectorStore):
        vecs = make_vecs(1)
        store.upsert("col-a", [1], vecs, ["t1"], ["f.md"])
        store.upsert("col-b", [2], vecs, ["t2"], ["g.md"])
        store.delete_collection("col-a")
        assert store.collection_exists("col-b") is True
        assert store.get_chunk_count("col-b") == 1


# ---------------------------------------------------------------------------
# FAISSVectorStore — get_chunk_count
# ---------------------------------------------------------------------------


class TestFAISSVectorStoreGetChunkCount:
    def test_zero_before_any_upsert(self, store: FAISSVectorStore):
        assert store.get_chunk_count("col") == 0

    def test_correct_count_after_upserts(self, store: FAISSVectorStore):
        store.upsert("col", [1, 2, 3], make_vecs(3), ["t"] * 3, ["f.md"] * 3)
        assert store.get_chunk_count("col") == 3

    def test_count_decreases_after_remove(self, store: FAISSVectorStore):
        store.upsert("col", [1, 2], make_vecs(2), ["t1", "t2"], ["f.md"] * 2)
        store.remove_ids("col", [1])
        assert store.get_chunk_count("col") == 1

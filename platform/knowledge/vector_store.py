"""IVectorStore interface, FAISSVectorStore, and internal ChunkStore."""

from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single result returned by a vector store query."""

    faiss_id: int
    text: str
    source_file: str
    score: float
    collection: str = ""


class IVectorStore(ABC):
    """Abstract interface for a vector store.

    Each collection is a named, independent index.
    The integer IDs used here correspond to KnowledgeChunkRow.id in SQLite,
    ensuring a single consistent ID across both stores.
    """

    @abstractmethod
    def upsert(
        self,
        collection: str,
        ids: list[int],
        vectors: list[list[float]],
        texts: list[str],
        source_files: list[str],
    ) -> None:
        """Add or update vectors in the given collection."""
        ...

    @abstractmethod
    def query(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
    ) -> list[SearchResult]:
        """Return the top_k nearest neighbours for the given query vector."""
        ...

    @abstractmethod
    def remove_ids(self, collection: str, ids: list[int]) -> None:
        """Remove specific chunk ids from the collection."""
        ...

    @abstractmethod
    def delete_collection(self, collection: str) -> None:
        """Delete the entire collection index and chunk store."""
        ...

    @abstractmethod
    def collection_exists(self, collection: str) -> bool:
        """Return True if an index file for the collection exists."""
        ...

    @abstractmethod
    def get_chunk_count(self, collection: str) -> int:
        """Return the number of vectors stored in the collection."""
        ...


# ---------------------------------------------------------------------------
# Internal ChunkStore
# ---------------------------------------------------------------------------


@dataclass
class _ChunkEntry:
    text: str
    source_file: str


class _ChunkStore:
    """Manages a JSON file mapping int id → {text, source_file}.

    Written atomically via a temp-file + os.replace pattern. Reading a missing
    or corrupt file returns an empty dict rather than raising.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> dict[int, _ChunkEntry]:
        if not self._path.exists():
            return {}
        try:
            with self._path.open(encoding="utf-8") as f:
                raw: dict[str, dict[str, str]] = json.load(f)
            return {
                int(k): _ChunkEntry(text=v["text"], source_file=v["source_file"])
                for k, v in raw.items()
            }
        except (json.JSONDecodeError, KeyError, ValueError):
            return {}

    def save(self, data: dict[int, _ChunkEntry]) -> None:
        """Write data atomically — readers never see a partial file."""
        payload = {
            str(id_): {"text": entry.text, "source_file": entry.source_file}
            for id_, entry in data.items()
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=self._path.parent, suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp_name, self._path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def upsert(self, entries: dict[int, _ChunkEntry]) -> None:
        """Merge new entries into the existing store."""
        data = self.load()
        data.update(entries)
        self.save(data)

    def remove_ids(self, ids: list[int]) -> None:
        """Remove the given ids from the store."""
        data = self.load()
        for id_ in ids:
            data.pop(id_, None)
        self.save(data)

    def clear(self) -> None:
        """Delete the JSON file if it exists."""
        if self._path.exists():
            self._path.unlink()


# ---------------------------------------------------------------------------
# FAISSVectorStore
# ---------------------------------------------------------------------------


class FAISSVectorStore(IVectorStore):
    """FAISS-backed vector store.

    Per-collection layout under base_path:
        {collection}.index        — FAISS IndexIDMap2(IndexFlatIP) binary
        {collection}.chunks.json  — ChunkStore JSON (id → text + source_file)

    The inner index uses IndexFlatIP (inner product). OpenAI embeddings
    should be L2-normalised before insertion so that IP equals cosine similarity.

    The SQLite KnowledgeChunkRow.id is used directly as the FAISS int64 id.
    """

    def __init__(self, base_path: Path, dimensions: int = 1536) -> None:
        self._base_path = base_path
        self._dimensions = dimensions
        base_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # IVectorStore
    # ------------------------------------------------------------------

    def upsert(
        self,
        collection: str,
        ids: list[int],
        vectors: list[list[float]],
        texts: list[str],
        source_files: list[str],
    ) -> None:
        if not ids:
            return
        index = self._load_or_create_index(collection)
        vecs_np = np.array(vectors, dtype=np.float32)
        ids_np = np.array(ids, dtype=np.int64)
        index.add_with_ids(vecs_np, ids_np)
        self._save_index(collection, index)

        entries = {
            id_: _ChunkEntry(text=text, source_file=sf)
            for id_, text, sf in zip(ids, texts, source_files)
        }
        self._chunk_store(collection).upsert(entries)

    def query(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
    ) -> list[SearchResult]:
        if not self.collection_exists(collection):
            return []
        index = faiss.read_index(str(self._index_path(collection)))
        chunk_data = self._chunk_store(collection).load()

        query_np = np.array([vector], dtype=np.float32)
        actual_k = min(top_k, index.ntotal)
        if actual_k == 0:
            return []
        scores, faiss_ids = index.search(query_np, actual_k)

        results: list[SearchResult] = []
        for score, fid in zip(scores[0], faiss_ids[0]):
            if fid == -1:
                continue
            entry = chunk_data.get(int(fid))
            if entry is None:
                continue
            results.append(
                SearchResult(
                    faiss_id=int(fid),
                    text=entry.text,
                    source_file=entry.source_file,
                    score=float(score),
                )
            )
        return results

    def remove_ids(self, collection: str, ids: list[int]) -> None:
        if not ids or not self.collection_exists(collection):
            return
        index = faiss.read_index(str(self._index_path(collection)))
        ids_np = np.array(ids, dtype=np.int64)
        selector = faiss.IDSelectorBatch(ids_np)
        index.remove_ids(selector)
        self._save_index(collection, index)
        self._chunk_store(collection).remove_ids(ids)

    def delete_collection(self, collection: str) -> None:
        path = self._index_path(collection)
        if path.exists():
            path.unlink()
        self._chunk_store(collection).clear()

    def collection_exists(self, collection: str) -> bool:
        return self._index_path(collection).exists()

    def get_chunk_count(self, collection: str) -> int:
        if not self.collection_exists(collection):
            return 0
        index = faiss.read_index(str(self._index_path(collection)))
        return index.ntotal

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _index_path(self, collection: str) -> Path:
        return self._base_path / f"{collection}.index"

    def _chunk_store(self, collection: str) -> _ChunkStore:
        return _ChunkStore(self._base_path / f"{collection}.chunks.json")

    def _load_or_create_index(self, collection: str) -> faiss.Index:
        path = self._index_path(collection)
        if path.exists():
            return faiss.read_index(str(path))
        inner = faiss.IndexFlatIP(self._dimensions)
        return faiss.IndexIDMap2(inner)

    def _save_index(self, collection: str, index: faiss.Index) -> None:
        faiss.write_index(index, str(self._index_path(collection)))

"""KnowledgeIndexer — ingestion pipeline: Document → Chunk → Embed → Store."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from platform.knowledge.chunker import TextChunker
from platform.knowledge.config import CollectionConfig, KnowledgeConfig
from platform.knowledge.embedder import IEmbedder
from platform.knowledge.loader import DocumentLoader, _TEXT_SUFFIXES
from platform.knowledge.vector_store import IVectorStore
from platform.persistence.repositories.knowledge_repo import KnowledgeRepository

_EMBED_BATCH_SIZE = 100


class KnowledgeIndexer:
    """Runs the ingestion pipeline for all configured knowledge collections.

    Incremental strategy (V2): one SHA-256 manifest per collection stores
    file-level hashes. If current hashes match the saved manifest, the
    collection is skipped. If any file changed, was added, or was removed,
    the entire collection is rebuilt from scratch.

    Manifest files: {vector_store_path}/manifests/{collection}.json
    """

    def __init__(
        self,
        config: KnowledgeConfig,
        embedder: IEmbedder,
        vector_store: IVectorStore,
        session_factory: sessionmaker[Session],
    ) -> None:
        self._config = config
        self._embedder = embedder
        self._vector_store = vector_store
        self._session_factory = session_factory
        self._loader = DocumentLoader()
        self._repo = KnowledgeRepository()
        self._manifests_dir = Path(config.vector_store_path) / "manifests"

    async def index_all(self) -> dict[str, int]:
        """Index all configured collections.

        Returns {collection_name: chunks_indexed}. A value of 0 means the
        collection was already up to date.
        """
        results: dict[str, int] = {}
        for col_cfg in self._config.collections:
            results[col_cfg.name] = await self._index_collection(col_cfg)
        return results

    async def index_collection(self, collection_name: str) -> int:
        """Index a single collection by name. Raises ValueError if unknown."""
        for col_cfg in self._config.collections:
            if col_cfg.name == collection_name:
                return await self._index_collection(col_cfg)
        raise ValueError(f"Unknown collection: '{collection_name}'")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    async def _index_collection(self, col_cfg: CollectionConfig) -> int:
        col_path = Path(col_cfg.path)
        manifest_path = self._manifests_dir / f"{col_cfg.name}.json"

        current_hashes = self._compute_file_hashes(col_path)
        saved_hashes = self._load_manifest(manifest_path)

        if current_hashes == saved_hashes:
            return 0

        return await self._rebuild_collection(
            col_cfg.name, col_path, current_hashes, manifest_path
        )

    async def _rebuild_collection(
        self,
        collection: str,
        col_path: Path,
        current_hashes: dict[str, str],
        manifest_path: Path,
    ) -> int:
        # 1. Clear existing vector store data first
        self._vector_store.delete_collection(collection)

        # 2. Clear SQLite metadata
        with self._session_factory() as session:
            self._repo.delete_by_collection(session, collection)
            session.commit()

        # 3. Load and chunk documents
        docs = self._loader.load_collection(col_path, collection) if col_path.is_dir() else []
        chunker = TextChunker(
            size=self._config.chunk_size, overlap=self._config.chunk_overlap
        )
        all_chunks = [chunk for doc in docs for chunk in chunker.chunk(doc)]

        if not all_chunks:
            self._save_manifest(manifest_path, current_hashes)
            return 0

        # 4. Embed all chunk texts (batched)
        texts = [c.content for c in all_chunks]
        vectors = await self._embed_in_batches(texts)

        # 5. Insert metadata into SQLite, collecting FAISS ids
        chunk_hashes = [
            hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts
        ]
        source_files = [c.metadata.get("source_file", "") for c in all_chunks]

        ids: list[int] = []
        with self._session_factory() as session:
            for chunk, chunk_hash in zip(all_chunks, chunk_hashes):
                id_ = self._repo.insert(
                    session,
                    collection=collection,
                    source_file=chunk.metadata.get("source_file", ""),
                    chunk_index=chunk.chunk_index,
                    chunk_hash=chunk_hash,
                )
                ids.append(id_)
            session.commit()

        # 6. Upsert vectors and chunk text into FAISS + ChunkStore
        self._vector_store.upsert(collection, ids, vectors, texts, source_files)

        # 7. Save manifest last — only reached on full success
        self._save_manifest(manifest_path, current_hashes)

        return len(all_chunks)

    async def _embed_in_batches(self, texts: list[str]) -> list[list[float]]:
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[i : i + _EMBED_BATCH_SIZE]
            all_vectors.extend(await self._embedder.embed(batch))
        return all_vectors

    def _compute_file_hashes(self, path: Path) -> dict[str, str]:
        """SHA-256 hash every file the loader would process under path."""
        hashes: dict[str, str] = {}
        if not path.is_dir():
            return hashes
        supported = _TEXT_SUFFIXES | {".pdf"}
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in supported:
                hashes[str(file_path)] = hashlib.sha256(
                    file_path.read_bytes()
                ).hexdigest()
        return hashes

    def _load_manifest(self, path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        try:
            with path.open(encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_manifest(self, path: Path, hashes: dict[str, str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(hashes, f, ensure_ascii=False, indent=2)
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

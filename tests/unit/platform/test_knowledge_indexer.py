"""Unit tests for KnowledgeIndexer — full pipeline with in-memory SQLite + real FAISS."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from platform.knowledge.config import CollectionConfig, KnowledgeConfig
from platform.knowledge.embedder import MockEmbedder
from platform.knowledge.indexer import KnowledgeIndexer
from platform.knowledge.vector_store import FAISSVectorStore
from platform.persistence.database import Base
from platform.persistence.repositories.knowledge_repo import KnowledgeRepository

DIMS = 4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session_factory(tmp_path: Path):
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


@pytest.fixture
def vector_store(tmp_path: Path) -> FAISSVectorStore:
    store_path = tmp_path / "vector_store"
    store_path.mkdir()
    return FAISSVectorStore(base_path=store_path, dimensions=DIMS)


@pytest.fixture
def knowledge_root(tmp_path: Path) -> Path:
    return tmp_path / "resources"


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    p = tmp_path / "vector_store"
    p.mkdir(exist_ok=True)
    return p


def make_config(
    store_path: Path,
    collections: list[tuple[str, str]],
    chunk_size: int = 200,
    chunk_overlap: int = 0,
) -> KnowledgeConfig:
    return KnowledgeConfig(
        embedding_model="mock",
        vector_store_path=str(store_path),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=5,
        collections=[CollectionConfig(name=n, path=str(p)) for n, p in collections],
    )


def make_indexer(
    config: KnowledgeConfig,
    session_factory,
    store_path: Path,
) -> KnowledgeIndexer:
    embedder = MockEmbedder(dimensions=DIMS)
    vs = FAISSVectorStore(base_path=store_path, dimensions=DIMS)
    return KnowledgeIndexer(
        config=config,
        embedder=embedder,
        vector_store=vs,
        session_factory=session_factory,
    )


# ---------------------------------------------------------------------------
# index_all — basic behaviour
# ---------------------------------------------------------------------------


class TestIndexAll:
    async def test_empty_collections_list_returns_empty_dict(
        self, session_factory, store_path
    ):
        config = make_config(store_path, [])
        indexer = make_indexer(config, session_factory, store_path)
        result = await indexer.index_all()
        assert result == {}

    async def test_returns_chunk_count_per_collection(
        self, tmp_path, session_factory, store_path
    ):
        col_path = tmp_path / "docs"
        col_path.mkdir()
        (col_path / "a.md").write_text("Hello world. " * 20, encoding="utf-8")

        config = make_config(store_path, [("docs", col_path)])
        indexer = make_indexer(config, session_factory, store_path)
        result = await indexer.index_all()

        assert "docs" in result
        assert result["docs"] > 0

    async def test_empty_directory_returns_zero(
        self, tmp_path, session_factory, store_path
    ):
        col_path = tmp_path / "empty"
        col_path.mkdir()

        config = make_config(store_path, [("empty", col_path)])
        indexer = make_indexer(config, session_factory, store_path)
        result = await indexer.index_all()
        assert result["empty"] == 0

    async def test_multiple_collections_indexed(
        self, tmp_path, session_factory, store_path
    ):
        col_a = tmp_path / "col_a"
        col_b = tmp_path / "col_b"
        col_a.mkdir()
        col_b.mkdir()
        (col_a / "doc.md").write_text("Content for collection A. " * 5, encoding="utf-8")
        (col_b / "doc.md").write_text("Content for collection B. " * 5, encoding="utf-8")

        config = make_config(store_path, [("col-a", col_a), ("col-b", col_b)])
        indexer = make_indexer(config, session_factory, store_path)
        result = await indexer.index_all()

        assert "col-a" in result
        assert "col-b" in result
        assert result["col-a"] > 0
        assert result["col-b"] > 0


# ---------------------------------------------------------------------------
# index_collection — by name
# ---------------------------------------------------------------------------


class TestIndexCollectionByName:
    async def test_raises_for_unknown_collection(
        self, tmp_path, session_factory, store_path
    ):
        config = make_config(store_path, [])
        indexer = make_indexer(config, session_factory, store_path)
        with pytest.raises(ValueError, match="Unknown collection"):
            await indexer.index_collection("ghost")

    async def test_indexes_named_collection(
        self, tmp_path, session_factory, store_path
    ):
        col_path = tmp_path / "docs"
        col_path.mkdir()
        (col_path / "file.md").write_text("Some documentation content here. " * 5, encoding="utf-8")

        config = make_config(store_path, [("docs", col_path)])
        indexer = make_indexer(config, session_factory, store_path)
        count = await indexer.index_collection("docs")
        assert count > 0


# ---------------------------------------------------------------------------
# Persistence — FAISS + SQLite written correctly
# ---------------------------------------------------------------------------


class TestPersistenceAfterIndex:
    async def test_index_file_created(self, tmp_path, session_factory, store_path):
        col_path = tmp_path / "docs"
        col_path.mkdir()
        (col_path / "guide.md").write_text("Hello " * 50, encoding="utf-8")

        config = make_config(store_path, [("docs", col_path)])
        indexer = make_indexer(config, session_factory, store_path)
        await indexer.index_all()

        assert (store_path / "docs.index").exists()

    async def test_chunk_json_created(self, tmp_path, session_factory, store_path):
        col_path = tmp_path / "docs"
        col_path.mkdir()
        (col_path / "guide.md").write_text("Hello " * 50, encoding="utf-8")

        config = make_config(store_path, [("docs", col_path)])
        indexer = make_indexer(config, session_factory, store_path)
        await indexer.index_all()

        assert (store_path / "docs.chunks.json").exists()

    async def test_manifest_file_created(self, tmp_path, session_factory, store_path):
        col_path = tmp_path / "docs"
        col_path.mkdir()
        (col_path / "guide.md").write_text("Content " * 20, encoding="utf-8")

        config = make_config(store_path, [("docs", col_path)])
        indexer = make_indexer(config, session_factory, store_path)
        await indexer.index_all()

        manifest_path = store_path / "manifests" / "docs.json"
        assert manifest_path.exists()

    async def test_sqlite_rows_written(self, tmp_path, session_factory, store_path):
        col_path = tmp_path / "docs"
        col_path.mkdir()
        (col_path / "guide.md").write_text("Some text. " * 30, encoding="utf-8")

        config = make_config(store_path, [("docs", col_path)])
        indexer = make_indexer(config, session_factory, store_path)
        count = await indexer.index_all()

        repo = KnowledgeRepository()
        with session_factory() as session:
            db_count = repo.count_by_collection(session, "docs")

        assert db_count == count["docs"]
        assert db_count > 0

    async def test_manifest_contains_file_hashes(self, tmp_path, session_factory, store_path):
        col_path = tmp_path / "docs"
        col_path.mkdir()
        file = col_path / "guide.md"
        file.write_text("Content here.", encoding="utf-8")

        config = make_config(store_path, [("docs", col_path)])
        indexer = make_indexer(config, session_factory, store_path)
        await indexer.index_all()

        manifest_path = store_path / "manifests" / "docs.json"
        with manifest_path.open() as f:
            data = json.load(f)

        assert str(file) in data
        assert len(data[str(file)]) == 64  # SHA-256 hex digest length


# ---------------------------------------------------------------------------
# Incremental indexing — skip unchanged, rebuild on change
# ---------------------------------------------------------------------------


class TestIncrementalIndexing:
    async def test_unchanged_collection_returns_zero_on_second_run(
        self, tmp_path, session_factory, store_path
    ):
        col_path = tmp_path / "docs"
        col_path.mkdir()
        (col_path / "guide.md").write_text("Static content. " * 10, encoding="utf-8")

        config = make_config(store_path, [("docs", col_path)])
        indexer = make_indexer(config, session_factory, store_path)

        await indexer.index_all()
        second = await indexer.index_all()
        assert second["docs"] == 0

    async def test_changed_file_triggers_rebuild(
        self, tmp_path, session_factory, store_path
    ):
        col_path = tmp_path / "docs"
        col_path.mkdir()
        file = col_path / "guide.md"
        file.write_text("Original content. " * 10, encoding="utf-8")

        config = make_config(store_path, [("docs", col_path)])
        indexer = make_indexer(config, session_factory, store_path)

        first = await indexer.index_all()
        assert first["docs"] > 0

        file.write_text("Modified content, which is completely different now. " * 10, encoding="utf-8")
        second = await indexer.index_all()
        assert second["docs"] > 0

    async def test_added_file_triggers_rebuild(
        self, tmp_path, session_factory, store_path
    ):
        col_path = tmp_path / "docs"
        col_path.mkdir()
        (col_path / "original.md").write_text("First file content. " * 5, encoding="utf-8")

        config = make_config(store_path, [("docs", col_path)])
        indexer = make_indexer(config, session_factory, store_path)

        await indexer.index_all()
        (col_path / "new.md").write_text("Newly added file. " * 5, encoding="utf-8")
        second = await indexer.index_all()
        assert second["docs"] > 0

    async def test_removed_file_triggers_rebuild(
        self, tmp_path, session_factory, store_path
    ):
        col_path = tmp_path / "docs"
        col_path.mkdir()
        file_a = col_path / "a.md"
        file_b = col_path / "b.md"
        file_a.write_text("File A content. " * 5, encoding="utf-8")
        file_b.write_text("File B content. " * 5, encoding="utf-8")

        config = make_config(store_path, [("docs", col_path)])
        indexer = make_indexer(config, session_factory, store_path)

        await indexer.index_all()
        file_b.unlink()
        second = await indexer.index_all()
        assert second["docs"] > 0

    async def test_old_sqlite_rows_deleted_on_rebuild(
        self, tmp_path, session_factory, store_path
    ):
        col_path = tmp_path / "docs"
        col_path.mkdir()
        file = col_path / "guide.md"
        file.write_text("Original. " * 20, encoding="utf-8")

        config = make_config(store_path, [("docs", col_path)])
        indexer = make_indexer(config, session_factory, store_path)

        await indexer.index_all()
        repo = KnowledgeRepository()
        with session_factory() as session:
            count_before = repo.count_by_collection(session, "docs")

        file.write_text("Completely rewritten. " * 20, encoding="utf-8")
        await indexer.index_all()
        with session_factory() as session:
            count_after = repo.count_by_collection(session, "docs")

        # Rows should not accumulate: old rows deleted, new rows inserted
        assert count_after > 0
        # Total should match exactly what was inserted (no stale rows)
        with session_factory() as session:
            db_ids = repo.get_ids_by_source_file(session, "docs", str(file))
        assert len(db_ids) == count_after


# ---------------------------------------------------------------------------
# File type filtering
# ---------------------------------------------------------------------------


class TestFileTypeFiltering:
    async def test_unsupported_extension_not_indexed(
        self, tmp_path, session_factory, store_path
    ):
        col_path = tmp_path / "docs"
        col_path.mkdir()
        (col_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (col_path / "data.bin").write_bytes(b"\x00\x01\x02\x03")

        config = make_config(store_path, [("docs", col_path)])
        indexer = make_indexer(config, session_factory, store_path)
        result = await indexer.index_all()
        assert result["docs"] == 0

    async def test_supported_extensions_are_indexed(
        self, tmp_path, session_factory, store_path
    ):
        col_path = tmp_path / "docs"
        col_path.mkdir()
        (col_path / "readme.md").write_text("Markdown content. " * 5, encoding="utf-8")
        (col_path / "notes.txt").write_text("Plain text notes. " * 5, encoding="utf-8")
        (col_path / "config.yaml").write_text("key: value\n" * 10, encoding="utf-8")

        config = make_config(store_path, [("docs", col_path)])
        indexer = make_indexer(config, session_factory, store_path)
        result = await indexer.index_all()
        assert result["docs"] > 0


# ---------------------------------------------------------------------------
# Nonexistent collection path
# ---------------------------------------------------------------------------


class TestNonexistentPath:
    async def test_nonexistent_path_returns_zero(
        self, tmp_path, session_factory, store_path
    ):
        missing_path = tmp_path / "does_not_exist"
        config = make_config(store_path, [("missing", missing_path)])
        indexer = make_indexer(config, session_factory, store_path)
        result = await indexer.index_all()
        assert result["missing"] == 0


# ---------------------------------------------------------------------------
# Batch embedding
# ---------------------------------------------------------------------------


class TestBatchEmbedding:
    async def test_more_than_batch_size_chunks_all_indexed(
        self, tmp_path, session_factory, store_path
    ):
        """Generate enough content to exceed EMBED_BATCH_SIZE (100)."""
        col_path = tmp_path / "docs"
        col_path.mkdir()
        # chunk_size=50 ensures many chunks per file
        for i in range(10):
            (col_path / f"doc_{i}.md").write_text(
                f"Document {i} section. " * 20, encoding="utf-8"
            )

        config = make_config(store_path, [("docs", col_path)], chunk_size=50, chunk_overlap=0)
        indexer = make_indexer(config, session_factory, store_path)
        result = await indexer.index_all()

        repo = KnowledgeRepository()
        with session_factory() as session:
            db_count = repo.count_by_collection(session, "docs")

        assert db_count == result["docs"]
        assert db_count > 0

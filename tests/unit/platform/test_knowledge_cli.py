"""Unit tests for scripts/index_knowledge.py CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from platform.knowledge.config import CollectionConfig, KnowledgeConfig
from platform.knowledge.embedder import MockEmbedder
from platform.knowledge.indexer import KnowledgeIndexer
from platform.knowledge.vector_store import FAISSVectorStore
from scripts.index_knowledge import _run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_indexer(results: dict[str, int]) -> MagicMock:
    indexer = MagicMock(spec=KnowledgeIndexer)
    indexer.index_all = AsyncMock(return_value=results)
    return indexer


# ---------------------------------------------------------------------------
# _run() — the core async logic
# ---------------------------------------------------------------------------


class TestRunCoreLogic:
    async def test_empty_results_prints_no_collections(self, capsys):
        indexer = _mock_indexer({})
        rc = await _run(indexer)
        assert rc == 0
        assert "No collections" in capsys.readouterr().out

    async def test_up_to_date_collection_prints_skip(self, capsys):
        indexer = _mock_indexer({"docs": 0})
        await _run(indexer)
        out = capsys.readouterr().out
        assert "up to date" in out or "skipped" in out

    async def test_indexed_collection_prints_chunk_count(self, capsys):
        indexer = _mock_indexer({"docs": 42})
        await _run(indexer)
        out = capsys.readouterr().out
        assert "42" in out

    async def test_calls_index_all(self):
        indexer = _mock_indexer({"docs": 5})
        await _run(indexer)
        indexer.index_all.assert_called_once()

    async def test_returns_zero_on_success(self):
        indexer = _mock_indexer({"docs": 5})
        rc = await _run(indexer)
        assert rc == 0

    async def test_multiple_collections_both_printed(self, capsys):
        indexer = _mock_indexer({"docs": 10, "api": 0})
        await _run(indexer)
        out = capsys.readouterr().out
        assert "docs" in out
        assert "api" in out

    async def test_prints_summary_totals(self, capsys):
        indexer = _mock_indexer({"docs": 10, "api": 20, "specs": 0})
        await _run(indexer)
        out = capsys.readouterr().out
        # 30 total chunks, 2 rebuilt, 1 skipped
        assert "30" in out

    async def test_prints_rebuilt_count(self, capsys):
        indexer = _mock_indexer({"docs": 5, "api": 0})
        await _run(indexer)
        out = capsys.readouterr().out
        # 1 rebuilt out of 2 total
        assert "1/2" in out


# ---------------------------------------------------------------------------
# Integration: _run with real KnowledgeIndexer + MockEmbedder
# ---------------------------------------------------------------------------


class TestRunWithRealComponents:
    async def test_indexes_new_collection(self, tmp_path: Path):
        col_path = tmp_path / "docs"
        col_path.mkdir()
        (col_path / "guide.md").write_text("Hello world. " * 20, encoding="utf-8")

        config = KnowledgeConfig(
            embedding_model="mock",
            vector_store_path=str(tmp_path / "vs"),
            chunk_size=200,
            chunk_overlap=0,
            collections=[CollectionConfig(name="docs", path=str(col_path))],
        )
        embedder = MockEmbedder(dimensions=4)
        vs = FAISSVectorStore(tmp_path / "vs", dimensions=4)

        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool
        from platform.persistence.database import Base

        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        sf = sessionmaker(bind=engine)

        indexer = KnowledgeIndexer(config, embedder, vs, sf)
        rc = await _run(indexer)
        assert rc == 0

    async def test_no_op_on_second_run(self, tmp_path: Path, capsys):
        col_path = tmp_path / "docs"
        col_path.mkdir()
        (col_path / "guide.md").write_text("Static content. " * 10, encoding="utf-8")

        config = KnowledgeConfig(
            embedding_model="mock",
            vector_store_path=str(tmp_path / "vs"),
            chunk_size=200,
            chunk_overlap=0,
            collections=[CollectionConfig(name="docs", path=str(col_path))],
        )
        embedder = MockEmbedder(dimensions=4)
        vs = FAISSVectorStore(tmp_path / "vs", dimensions=4)

        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool
        from platform.persistence.database import Base

        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        sf = sessionmaker(bind=engine)

        indexer = KnowledgeIndexer(config, embedder, vs, sf)
        await _run(indexer)      # first run — indexes
        capsys.readouterr()      # clear output

        rc = await _run(indexer)  # second run — should skip
        out = capsys.readouterr().out
        assert rc == 0
        assert "skipped" in out or "up to date" in out


# ---------------------------------------------------------------------------
# main() error paths (tested without invoking subprocess)
# ---------------------------------------------------------------------------


class TestMainErrorPaths:
    def test_exits_1_when_config_missing(self, tmp_path: Path, monkeypatch):
        import os
        monkeypatch.chdir(tmp_path)  # no knowledge_config.yaml here
        from scripts.index_knowledge import main
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_exits_1_when_no_api_key(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        (tmp_path / "knowledge_config.yaml").write_text(
            "knowledge:\n  collections: [{name: docs, path: docs}]\n",
            encoding="utf-8",
        )
        from scripts.index_knowledge import main
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

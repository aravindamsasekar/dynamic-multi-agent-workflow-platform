"""Unit tests for knowledge startup wiring and startup indexing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.dependencies import _build_knowledge_stack, _index_with_summary, run_startup_indexing
from platform.knowledge.indexer import KnowledgeIndexer
from platform.knowledge.service import KnowledgeService
from platform.persistence.database import Base

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_VALID_CONFIG = """\
knowledge:
  embedding:
    model: text-embedding-3-small
  vector_store:
    path: data/knowledge
  chunking:
    size: 1000
    overlap: 200
  retrieval:
    top_k: 5
  collections:
    - name: docs
      path: resources/knowledge/docs
"""

_INVALID_CONFIG = "knowledge: !!python/object:builtins.dict {}"


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


# ---------------------------------------------------------------------------
# _build_knowledge_stack
# ---------------------------------------------------------------------------


class TestBuildKnowledgeStack:
    def test_missing_config_returns_none_tuple(self, tmp_path: Path, session_factory, capsys):
        service, indexer = _build_knowledge_stack(tmp_path / "missing.yaml", session_factory)
        assert service is None
        assert indexer is None

    def test_missing_config_logs_warning(self, tmp_path: Path, session_factory, capsys):
        _build_knowledge_stack(tmp_path / "missing.yaml", session_factory)
        assert "WARNING" in capsys.readouterr().err

    def test_missing_config_warning_mentions_filename(self, tmp_path: Path, session_factory, capsys):
        _build_knowledge_stack(tmp_path / "knowledge_config.yaml", session_factory)
        assert "knowledge_config.yaml" in capsys.readouterr().err

    def test_valid_config_returns_service(self, tmp_path: Path, session_factory, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        cfg = tmp_path / "knowledge_config.yaml"
        cfg.write_text(_VALID_CONFIG, encoding="utf-8")
        service, indexer = _build_knowledge_stack(cfg, session_factory)
        assert isinstance(service, KnowledgeService)

    def test_valid_config_returns_indexer(self, tmp_path: Path, session_factory, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        cfg = tmp_path / "knowledge_config.yaml"
        cfg.write_text(_VALID_CONFIG, encoding="utf-8")
        service, indexer = _build_knowledge_stack(cfg, session_factory)
        assert isinstance(indexer, KnowledgeIndexer)

    def test_bad_yaml_returns_none_and_logs(self, tmp_path: Path, session_factory, capsys):
        cfg = tmp_path / "knowledge_config.yaml"
        cfg.write_text("{not valid yaml: [", encoding="utf-8")
        service, indexer = _build_knowledge_stack(cfg, session_factory)
        assert service is None
        assert indexer is None
        assert "WARNING" in capsys.readouterr().err

    def test_missing_api_key_returns_none(self, tmp_path: Path, session_factory, monkeypatch, capsys):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = tmp_path / "knowledge_config.yaml"
        cfg.write_text(_VALID_CONFIG, encoding="utf-8")
        service, indexer = _build_knowledge_stack(cfg, session_factory)
        assert service is None
        assert indexer is None

    def test_empty_collections_config_still_returns_service(
        self, tmp_path: Path, session_factory, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        cfg = tmp_path / "knowledge_config.yaml"
        cfg.write_text("knowledge:\n  collections: []\n", encoding="utf-8")
        service, indexer = _build_knowledge_stack(cfg, session_factory)
        assert isinstance(service, KnowledgeService)


# ---------------------------------------------------------------------------
# _index_with_summary
# ---------------------------------------------------------------------------


class TestIndexWithSummary:
    async def test_calls_index_all(self):
        indexer = MagicMock(spec=KnowledgeIndexer)
        indexer.index_all = AsyncMock(return_value={})
        await _index_with_summary(indexer)
        indexer.index_all.assert_called_once()

    async def test_returns_results_dict(self):
        indexer = MagicMock(spec=KnowledgeIndexer)
        indexer.index_all = AsyncMock(return_value={"docs": 10, "arch": 0})
        result = await _index_with_summary(indexer)
        assert result == {"docs": 10, "arch": 0}

    async def test_logs_summary(self, capsys):
        indexer = MagicMock(spec=KnowledgeIndexer)
        indexer.index_all = AsyncMock(return_value={"docs": 5, "api": 0})
        await _index_with_summary(indexer)
        assert "Knowledge" in capsys.readouterr().err

    async def test_returns_empty_dict_on_exception(self, capsys):
        indexer = MagicMock(spec=KnowledgeIndexer)
        indexer.index_all = AsyncMock(side_effect=RuntimeError("FAISS error"))
        result = await _index_with_summary(indexer)
        assert result == {}
        assert "WARNING" in capsys.readouterr().err

    async def test_skipped_collection_not_counted_in_rebuilt(self, capsys):
        indexer = MagicMock(spec=KnowledgeIndexer)
        indexer.index_all = AsyncMock(return_value={"docs": 0, "api": 15})
        await _index_with_summary(indexer)
        err = capsys.readouterr().err
        # 1 rebuilt (api), 1 skipped (docs)
        assert "1" in err


# ---------------------------------------------------------------------------
# run_startup_indexing (uses global state — test via monkeypatching)
# ---------------------------------------------------------------------------


class TestRunStartupIndexing:
    async def test_returns_empty_dict_when_no_indexer(self, monkeypatch):
        import api.dependencies as deps
        monkeypatch.setattr(deps, "_knowledge_indexer", None)
        result = await run_startup_indexing()
        assert result == {}

    async def test_delegates_to_indexer_when_present(self, monkeypatch):
        import api.dependencies as deps
        mock_indexer = MagicMock(spec=KnowledgeIndexer)
        mock_indexer.index_all = AsyncMock(return_value={"docs": 42})
        monkeypatch.setattr(deps, "_knowledge_indexer", mock_indexer)
        result = await run_startup_indexing()
        assert result == {"docs": 42}
        mock_indexer.index_all.assert_called_once()


# ---------------------------------------------------------------------------
# Startup with missing resources/knowledge path
# ---------------------------------------------------------------------------


class TestStartupWithMissingResources:
    async def test_missing_collection_path_does_not_raise(
        self, tmp_path: Path, session_factory, monkeypatch
    ):
        """If resources/knowledge/docs doesn't exist, indexer returns 0 (skip)."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        cfg = tmp_path / "knowledge_config.yaml"
        # Point to a non-existent path
        cfg.write_text(
            f"knowledge:\n  collections:\n    - name: docs\n      path: {tmp_path / 'nonexistent'}\n",
            encoding="utf-8",
        )
        service, indexer = _build_knowledge_stack(cfg, session_factory)
        assert service is not None
        assert indexer is not None
        # index_all should return 0 for the missing path (manifest == hashes == {})
        results = await indexer.index_all()
        assert results == {"docs": 0}

    async def test_empty_collection_directory_returns_zero(
        self, tmp_path: Path, session_factory, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        col_path = tmp_path / "docs"
        col_path.mkdir()
        cfg = tmp_path / "knowledge_config.yaml"
        cfg.write_text(
            f"knowledge:\n  collections:\n    - name: docs\n      path: {col_path}\n",
            encoding="utf-8",
        )
        service, indexer = _build_knowledge_stack(cfg, session_factory)
        results = await indexer.index_all()
        assert results["docs"] == 0

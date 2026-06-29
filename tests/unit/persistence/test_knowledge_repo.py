"""Unit tests for KnowledgeRepository — runs against in-memory SQLite."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from platform.persistence.database import Base
from platform.persistence.repositories.knowledge_repo import KnowledgeRepository


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as s:
        yield s


def _insert(session, collection="col-a", source_file="f.md", chunk_index=0, chunk_hash="abc"):
    repo = KnowledgeRepository()
    return repo.insert(
        session,
        collection=collection,
        source_file=source_file,
        chunk_index=chunk_index,
        chunk_hash=chunk_hash,
    )


# ---------------------------------------------------------------------------
# insert
# ---------------------------------------------------------------------------


class TestKnowledgeRepositoryInsert:
    def test_returns_positive_integer_id(self, session):
        id_ = _insert(session)
        assert isinstance(id_, int)
        assert id_ > 0

    def test_successive_inserts_return_distinct_ids(self, session):
        id1 = _insert(session, chunk_index=0)
        id2 = _insert(session, chunk_index=1)
        assert id1 != id2

    def test_ids_are_autoincrement(self, session):
        id1 = _insert(session, chunk_index=0)
        id2 = _insert(session, chunk_index=1)
        assert id2 == id1 + 1

    def test_row_accessible_after_insert(self, session):
        from platform.persistence.models import KnowledgeChunkRow
        id_ = _insert(session, collection="my-col", source_file="guide.md", chunk_index=3, chunk_hash="h123")
        session.commit()
        row = session.get(KnowledgeChunkRow, id_)
        assert row is not None
        assert row.collection == "my-col"
        assert row.source_file == "guide.md"
        assert row.chunk_index == 3
        assert row.chunk_hash == "h123"


# ---------------------------------------------------------------------------
# get_ids_by_source_file
# ---------------------------------------------------------------------------


class TestGetIdsBySourceFile:
    def test_returns_ids_for_matching_file(self, session):
        repo = KnowledgeRepository()
        id1 = _insert(session, source_file="a.md", chunk_index=0)
        id2 = _insert(session, source_file="a.md", chunk_index=1)
        _insert(session, source_file="b.md", chunk_index=0)

        ids = repo.get_ids_by_source_file(session, "col-a", "a.md")
        assert sorted(ids) == sorted([id1, id2])

    def test_returns_empty_for_unknown_file(self, session):
        repo = KnowledgeRepository()
        assert repo.get_ids_by_source_file(session, "col-a", "missing.md") == []

    def test_scoped_to_collection(self, session):
        repo = KnowledgeRepository()
        _insert(session, collection="col-a", source_file="x.md")
        _insert(session, collection="col-b", source_file="x.md")

        ids_a = repo.get_ids_by_source_file(session, "col-a", "x.md")
        ids_b = repo.get_ids_by_source_file(session, "col-b", "x.md")
        assert len(ids_a) == 1
        assert len(ids_b) == 1
        assert ids_a[0] != ids_b[0]


# ---------------------------------------------------------------------------
# delete_by_source_file
# ---------------------------------------------------------------------------


class TestDeleteBySourceFile:
    def test_deletes_rows_and_returns_ids(self, session):
        repo = KnowledgeRepository()
        id1 = _insert(session, source_file="a.md", chunk_index=0)
        id2 = _insert(session, source_file="a.md", chunk_index=1)

        deleted = repo.delete_by_source_file(session, "col-a", "a.md")
        session.commit()
        assert sorted(deleted) == sorted([id1, id2])
        assert repo.get_ids_by_source_file(session, "col-a", "a.md") == []

    def test_does_not_delete_other_files(self, session):
        repo = KnowledgeRepository()
        _insert(session, source_file="a.md")
        id_b = _insert(session, source_file="b.md")

        repo.delete_by_source_file(session, "col-a", "a.md")
        session.commit()
        remaining = repo.get_ids_by_source_file(session, "col-a", "b.md")
        assert remaining == [id_b]

    def test_returns_empty_list_for_unknown_file(self, session):
        repo = KnowledgeRepository()
        assert repo.delete_by_source_file(session, "col-a", "ghost.md") == []

    def test_scoped_to_collection(self, session):
        repo = KnowledgeRepository()
        _insert(session, collection="col-a", source_file="x.md")
        id_b = _insert(session, collection="col-b", source_file="x.md")

        repo.delete_by_source_file(session, "col-a", "x.md")
        session.commit()
        assert repo.get_ids_by_source_file(session, "col-b", "x.md") == [id_b]


# ---------------------------------------------------------------------------
# delete_by_collection
# ---------------------------------------------------------------------------


class TestDeleteByCollection:
    def test_deletes_all_rows_in_collection(self, session):
        repo = KnowledgeRepository()
        id1 = _insert(session, chunk_index=0)
        id2 = _insert(session, chunk_index=1)

        deleted = repo.delete_by_collection(session, "col-a")
        session.commit()
        assert sorted(deleted) == sorted([id1, id2])
        assert repo.count_by_collection(session, "col-a") == 0

    def test_does_not_touch_other_collections(self, session):
        repo = KnowledgeRepository()
        _insert(session, collection="col-a")
        id_b = _insert(session, collection="col-b")

        repo.delete_by_collection(session, "col-a")
        session.commit()
        assert repo.count_by_collection(session, "col-b") == 1
        assert repo.get_ids_by_source_file(session, "col-b", "f.md") == [id_b]

    def test_returns_empty_list_for_unknown_collection(self, session):
        repo = KnowledgeRepository()
        assert repo.delete_by_collection(session, "ghost-col") == []


# ---------------------------------------------------------------------------
# count_by_collection
# ---------------------------------------------------------------------------


class TestCountByCollection:
    def test_zero_for_empty_collection(self, session):
        assert KnowledgeRepository().count_by_collection(session, "col-a") == 0

    def test_counts_correctly(self, session):
        repo = KnowledgeRepository()
        _insert(session, chunk_index=0)
        _insert(session, chunk_index=1)
        _insert(session, chunk_index=2)
        assert repo.count_by_collection(session, "col-a") == 3

    def test_scoped_to_collection(self, session):
        repo = KnowledgeRepository()
        _insert(session, collection="col-a", chunk_index=0)
        _insert(session, collection="col-a", chunk_index=1)
        _insert(session, collection="col-b", chunk_index=0)
        assert repo.count_by_collection(session, "col-a") == 2
        assert repo.count_by_collection(session, "col-b") == 1


# ---------------------------------------------------------------------------
# list_source_files
# ---------------------------------------------------------------------------


class TestListSourceFiles:
    def test_returns_distinct_source_files(self, session):
        repo = KnowledgeRepository()
        _insert(session, source_file="a.md", chunk_index=0)
        _insert(session, source_file="a.md", chunk_index=1)
        _insert(session, source_file="b.md", chunk_index=0)

        files = repo.list_source_files(session, "col-a")
        assert sorted(files) == ["a.md", "b.md"]

    def test_empty_for_unknown_collection(self, session):
        assert KnowledgeRepository().list_source_files(session, "ghost") == []

    def test_scoped_to_collection(self, session):
        repo = KnowledgeRepository()
        _insert(session, collection="col-a", source_file="x.md")
        _insert(session, collection="col-b", source_file="y.md")

        assert repo.list_source_files(session, "col-a") == ["x.md"]
        assert repo.list_source_files(session, "col-b") == ["y.md"]


# ---------------------------------------------------------------------------
# collection_exists
# ---------------------------------------------------------------------------


class TestCollectionExists:
    def test_false_for_empty_db(self, session):
        assert KnowledgeRepository().collection_exists(session, "col-a") is False

    def test_true_after_insert(self, session):
        _insert(session)
        assert KnowledgeRepository().collection_exists(session, "col-a") is True

    def test_false_after_delete_all(self, session):
        repo = KnowledgeRepository()
        _insert(session)
        repo.delete_by_collection(session, "col-a")
        session.commit()
        assert repo.collection_exists(session, "col-a") is False


# ---------------------------------------------------------------------------
# list_collections
# ---------------------------------------------------------------------------


class TestListCollections:
    def test_empty_when_no_rows(self, session):
        assert KnowledgeRepository().list_collections(session) == []

    def test_returns_distinct_collections(self, session):
        repo = KnowledgeRepository()
        _insert(session, collection="col-a")
        _insert(session, collection="col-a")
        _insert(session, collection="col-b")

        cols = repo.list_collections(session)
        assert sorted(cols) == ["col-a", "col-b"]

    def test_sorted_alphabetically(self, session):
        repo = KnowledgeRepository()
        _insert(session, collection="zebra")
        _insert(session, collection="alpha")
        _insert(session, collection="middle")

        assert repo.list_collections(session) == ["alpha", "middle", "zebra"]


# ---------------------------------------------------------------------------
# count_chunks_per_file
# ---------------------------------------------------------------------------


class TestCountChunksPerFile:
    def test_empty_for_unknown_collection(self, session):
        assert KnowledgeRepository().count_chunks_per_file(session, "ghost") == {}

    def test_counts_per_file(self, session):
        repo = KnowledgeRepository()
        _insert(session, source_file="a.md", chunk_index=0)
        _insert(session, source_file="a.md", chunk_index=1)
        _insert(session, source_file="b.md", chunk_index=0)

        counts = repo.count_chunks_per_file(session, "col-a")
        assert counts == {"a.md": 2, "b.md": 1}

    def test_scoped_to_collection(self, session):
        repo = KnowledgeRepository()
        _insert(session, collection="col-a", source_file="x.md")
        _insert(session, collection="col-b", source_file="x.md")
        _insert(session, collection="col-b", source_file="x.md")

        counts_a = repo.count_chunks_per_file(session, "col-a")
        counts_b = repo.count_chunks_per_file(session, "col-b")
        assert counts_a == {"x.md": 1}
        assert counts_b == {"x.md": 2}

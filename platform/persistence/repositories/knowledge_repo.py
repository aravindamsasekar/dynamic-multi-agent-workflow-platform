"""KnowledgeRepository — CRUD for the knowledge_chunks table."""

from __future__ import annotations

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from platform.persistence.models import KnowledgeChunkRow


class KnowledgeRepository:
    """Manages KnowledgeChunkRow records.

    Every method receives a Session from the caller; this class does not own
    the session lifecycle or transaction boundaries.
    """

    def insert(
        self,
        session: Session,
        collection: str,
        source_file: str,
        chunk_index: int,
        chunk_hash: str,
    ) -> int:
        """Insert a chunk metadata row and return its auto-generated id.

        The returned id is used directly as the FAISS vector id.
        """
        row = KnowledgeChunkRow(
            collection=collection,
            source_file=source_file,
            chunk_index=chunk_index,
            chunk_hash=chunk_hash,
        )
        session.add(row)
        session.flush()  # populate row.id without committing
        return row.id

    def get_ids_by_source_file(
        self,
        session: Session,
        collection: str,
        source_file: str,
    ) -> list[int]:
        """Return chunk ids belonging to a specific source file in a collection."""
        stmt = select(KnowledgeChunkRow.id).where(
            KnowledgeChunkRow.collection == collection,
            KnowledgeChunkRow.source_file == source_file,
        )
        return list(session.scalars(stmt))

    def delete_by_source_file(
        self,
        session: Session,
        collection: str,
        source_file: str,
    ) -> list[int]:
        """Delete all chunks for a source file and return the deleted ids."""
        ids = self.get_ids_by_source_file(session, collection, source_file)
        if ids:
            session.execute(
                delete(KnowledgeChunkRow).where(
                    KnowledgeChunkRow.collection == collection,
                    KnowledgeChunkRow.source_file == source_file,
                )
            )
        return ids

    def delete_by_collection(
        self,
        session: Session,
        collection: str,
    ) -> list[int]:
        """Delete all chunks for a collection and return the deleted ids."""
        stmt = select(KnowledgeChunkRow.id).where(
            KnowledgeChunkRow.collection == collection
        )
        ids = list(session.scalars(stmt))
        if ids:
            session.execute(
                delete(KnowledgeChunkRow).where(
                    KnowledgeChunkRow.collection == collection
                )
            )
        return ids

    def count_by_collection(self, session: Session, collection: str) -> int:
        """Return the total number of chunks in a collection."""
        stmt = select(func.count()).where(KnowledgeChunkRow.collection == collection)
        return session.scalar(stmt) or 0

    def list_source_files(self, session: Session, collection: str) -> list[str]:
        """Return distinct source file paths for a collection, sorted."""
        stmt = (
            select(KnowledgeChunkRow.source_file)
            .where(KnowledgeChunkRow.collection == collection)
            .distinct()
            .order_by(KnowledgeChunkRow.source_file)
        )
        return list(session.scalars(stmt))

    def collection_exists(self, session: Session, collection: str) -> bool:
        """Return True if any chunks are indexed for the collection."""
        stmt = select(func.count()).where(KnowledgeChunkRow.collection == collection)
        return (session.scalar(stmt) or 0) > 0

    def list_collections(self, session: Session) -> list[str]:
        """Return all distinct collection names, sorted."""
        stmt = (
            select(KnowledgeChunkRow.collection)
            .distinct()
            .order_by(KnowledgeChunkRow.collection)
        )
        return list(session.scalars(stmt))

    def get_chunk_indices_by_ids(
        self, session: Session, ids: list[int]
    ) -> dict[int, int]:
        """Return {id: chunk_index} for a batch of chunk ids."""
        if not ids:
            return {}
        stmt = select(KnowledgeChunkRow.id, KnowledgeChunkRow.chunk_index).where(
            KnowledgeChunkRow.id.in_(ids)
        )
        return {row.id: row.chunk_index for row in session.execute(stmt)}

    def count_chunks_per_file(
        self, session: Session, collection: str
    ) -> dict[str, int]:
        """Return a mapping of source_file → chunk count for a collection."""
        stmt = (
            select(KnowledgeChunkRow.source_file, func.count().label("cnt"))
            .where(KnowledgeChunkRow.collection == collection)
            .group_by(KnowledgeChunkRow.source_file)
        )
        return {row.source_file: row.cnt for row in session.execute(stmt)}

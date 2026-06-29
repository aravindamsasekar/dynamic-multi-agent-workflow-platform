"""Read-only knowledge endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import get_db_session, get_knowledge_service
from api.schemas.knowledge import (
    CollectionDetailResponse,
    CollectionSummaryResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from platform.knowledge.service import KnowledgeService
from platform.persistence.repositories.knowledge_repo import KnowledgeRepository

router = APIRouter()

_repo = KnowledgeRepository()

_503 = HTTPException(
    status_code=503,
    detail="Knowledge service is not configured. Add knowledge_config.yaml and restart.",
)


@router.get("/collections", response_model=list[CollectionSummaryResponse])
def list_collections(
    session: Session = Depends(get_db_session),
    ks: KnowledgeService | None = Depends(get_knowledge_service),
) -> list[CollectionSummaryResponse]:
    if ks is None:
        raise _503
    names = _repo.list_collections(session)
    result: list[CollectionSummaryResponse] = []
    for name in names:
        docs = _repo.list_source_files(session, name)
        chunks = _repo.count_by_collection(session, name)
        result.append(
            CollectionSummaryResponse(
                name=name,
                document_count=len(docs),
                chunk_count=chunks,
            )
        )
    return result


@router.get("/collections/{collection}", response_model=CollectionDetailResponse)
def get_collection(
    collection: str,
    session: Session = Depends(get_db_session),
    ks: KnowledgeService | None = Depends(get_knowledge_service),
) -> CollectionDetailResponse:
    if ks is None:
        raise _503
    if not _repo.collection_exists(session, collection):
        raise HTTPException(status_code=404, detail=f"Collection '{collection}' not found")
    docs = _repo.list_source_files(session, collection)
    chunks = _repo.count_by_collection(session, collection)
    return CollectionDetailResponse(name=collection, chunk_count=chunks, documents=docs)


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    session: Session = Depends(get_db_session),
    ks: KnowledgeService | None = Depends(get_knowledge_service),
) -> SearchResponse:
    if ks is None:
        raise _503

    results = await ks.search(request.query, request.collections, request.top_k)

    faiss_ids = [r.faiss_id for r in results]
    chunk_index_map = _repo.get_chunk_indices_by_ids(session, faiss_ids)

    items = [
        SearchResultItem(
            score=r.score,
            collection=r.collection,
            source_file=r.source_file,
            chunk_index=chunk_index_map.get(r.faiss_id, -1),
            text=r.text,
        )
        for r in results
    ]
    return SearchResponse(query=request.query, results=items)

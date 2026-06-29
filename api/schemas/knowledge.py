"""Knowledge API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CollectionSummaryResponse(BaseModel):
    name: str
    document_count: int
    chunk_count: int


class CollectionDetailResponse(BaseModel):
    name: str
    chunk_count: int
    documents: list[str]


class SearchRequest(BaseModel):
    query: str
    collections: list[str]
    top_k: int = Field(default=5, ge=1)


class SearchResultItem(BaseModel):
    score: float
    collection: str
    source_file: str
    chunk_index: int
    text: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]

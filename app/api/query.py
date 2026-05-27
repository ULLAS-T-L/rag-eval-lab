from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agents.router import ReasoningRouter
from app.db.session import get_db
from app.retrieval.embeddings import PlaceholderEmbeddingProvider
from app.retrieval.rerankers import NoOpReranker
from app.retrieval.retriever import RetrievalFilters, RetrievalService, VectorRetriever

router = APIRouter(prefix="/query", tags=["query"])


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: Optional[str] = None
    source_file: Optional[str] = None
    company: Optional[str] = None
    year: Optional[int] = None
    page: Optional[int] = Field(default=None, ge=1)
    page_start: Optional[int] = Field(default=None, ge=1)
    page_end: Optional[int] = Field(default=None, ge=1)
    section_title: Optional[str] = None


class RetrieveResponse(BaseModel):
    chunks: list[dict[str, Any]]
    latency_ms: int
    retrieval_strategy: str


class AskFilters(BaseModel):
    source_file: Optional[str] = None
    company: Optional[str] = None
    year: Optional[int] = None
    section_title: Optional[str] = None
    page_start: Optional[int] = Field(default=None, ge=1)
    page_end: Optional[int] = Field(default=None, ge=1)


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    filters: AskFilters = Field(default_factory=AskFilters)


class AskResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]
    applied_filters: dict[str, Any]
    retrieval_strategy: str
    confidence_score: Optional[float] = None
    latency_ms: int
    token_usage: Optional[dict[str, Any]] = None


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest, db: Session = Depends(get_db)) -> RetrieveResponse:
    retriever = VectorRetriever(
        session=db,
        embedding_provider=PlaceholderEmbeddingProvider(),
    )
    service = RetrievalService(session=db, retriever=retriever, reranker=NoOpReranker())
    filters = RetrievalFilters(
        document_id=request.document_id,
        source_file=request.source_file,
        company=request.company,
        year=request.year,
        page=request.page,
        page_start=request.page_start,
        page_end=request.page_end,
        section_title=request.section_title,
    )
    chunks, latency_ms = service.retrieve(
        query=request.query,
        top_k=request.top_k,
        filters=filters,
    )
    return RetrieveResponse(
        chunks=[
            {
                "chunk_id": chunk.chunk_id,
                "chunk_text": chunk.chunk_text,
                "similarity_score": chunk.similarity_score,
                "document_metadata": chunk.document_metadata,
                "citations": [citation.__dict__ for citation in chunk.citations],
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ],
        latency_ms=latency_ms,
        retrieval_strategy=retriever.retrieval_strategy,
    )


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, db: Session = Depends(get_db)) -> AskResponse:
    router_service = ReasoningRouter(
        session=db,
        embedding_provider=PlaceholderEmbeddingProvider(),
        reranker=NoOpReranker(),
    )
    result = router_service.ask(
        query=request.query,
        top_k=request.top_k,
        filters=request.filters.model_dump(),
    )
    return AskResponse(
        answer=result.answer,
        citations=result.citations,
        retrieved_chunks=result.retrieved_chunks,
        applied_filters=result.applied_filters,
        retrieval_strategy=result.retrieval_strategy,
        confidence_score=result.confidence_score,
        latency_ms=result.latency_ms,
        token_usage=result.token_usage,
    )

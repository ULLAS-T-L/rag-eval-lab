from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

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
    page: Optional[int] = Field(default=None, ge=1)
    section_title: Optional[str] = None


class RetrieveResponse(BaseModel):
    chunks: list[dict[str, Any]]
    latency_ms: int
    retrieval_strategy: str


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
        page=request.page,
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

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Any, Optional, Protocol
from uuid import UUID

from sqlalchemy import Select, and_, select, text
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document, Embedding, RetrievalLog
from app.retrieval.embeddings import EmbeddingProvider


@dataclass(frozen=True)
class RetrievalFilters:
    document_id: Optional[str] = None
    source_file: Optional[str] = None
    page: Optional[int] = None
    section_title: Optional[str] = None


@dataclass(frozen=True)
class Citation:
    document_id: str
    source_file: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    section_title: Optional[str]
    chunk_id: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    chunk_text: str
    similarity_score: float
    document_metadata: dict[str, Any]
    citations: list[Citation]
    metadata: dict

    @property
    def content(self) -> str:
        return self.chunk_text

    @property
    def score(self) -> float:
        return self.similarity_score


class Retriever(Protocol):
    retrieval_strategy: str

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[RetrievalFilters] = None,
    ) -> list[RetrievedChunk]:
        ...


class VectorRetriever:
    retrieval_strategy = "vector"

    def __init__(self, session: Session, embedding_provider: EmbeddingProvider) -> None:
        self.session = session
        self.embedding_provider = embedding_provider

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[RetrievalFilters] = None,
    ) -> list[RetrievedChunk]:
        query_vector = self.embedding_provider.embed_query(query)
        statement = self.build_query(query_vector=query_vector, top_k=top_k, filters=filters)
        rows = self.session.execute(statement).all()
        return [self._row_to_chunk(row) for row in rows]

    def build_query(
        self,
        *,
        query_vector: list[float],
        top_k: int,
        filters: Optional[RetrievalFilters] = None,
    ) -> Select:
        distance = Embedding.vector.cosine_distance(query_vector).label("distance")
        statement = (
            select(Chunk, Document, distance)
            .join(Embedding, Embedding.chunk_id == Chunk.id)
            .join(Document, Document.id == Chunk.document_id)
            .order_by(distance.asc())
            .limit(top_k)
        )

        predicates = self._filter_predicates(filters)
        if predicates:
            statement = statement.where(and_(*predicates))
        return statement

    def _filter_predicates(self, filters: Optional[RetrievalFilters]) -> list[Any]:
        if filters is None:
            return []

        predicates: list[Any] = []
        if filters.document_id:
            predicates.append(Chunk.document_id == UUID(filters.document_id))
        if filters.source_file:
            predicates.append(Chunk.chunk_metadata["source_file"].as_string() == filters.source_file)
        if filters.page is not None:
            predicates.append(Chunk.chunk_metadata["page_start"].as_integer() <= filters.page)
            predicates.append(Chunk.chunk_metadata["page_end"].as_integer() >= filters.page)
        if filters.section_title:
            predicates.append(Chunk.section_path == filters.section_title)
        return predicates

    def _row_to_chunk(self, row: Any) -> RetrievedChunk:
        chunk: Chunk = row[0]
        document: Document = row[1]
        distance = float(row[2])
        similarity_score = 1.0 - distance if isfinite(distance) else 0.0
        metadata = dict(chunk.chunk_metadata or {})
        citation = Citation(
            document_id=str(document.id),
            source_file=metadata.get("source_file"),
            page_start=metadata.get("page_start"),
            page_end=metadata.get("page_end"),
            section_title=chunk.section_path,
            chunk_id=str(chunk.id),
        )
        return RetrievedChunk(
            chunk_id=str(chunk.id),
            chunk_text=chunk.content,
            similarity_score=similarity_score,
            document_metadata=dict(document.doc_metadata or {}),
            citations=[citation],
            metadata=metadata,
        )


class HybridRetriever:
    retrieval_strategy = "hybrid"

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[RetrievalFilters] = None,
    ) -> list[RetrievedChunk]:
        return []


class RetrievalService:
    def __init__(self, session: Session, retriever: Retriever, reranker: Any) -> None:
        self.session = session
        self.retriever = retriever
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[RetrievalFilters] = None,
    ) -> tuple[list[RetrievedChunk], int]:
        start = perf_counter()
        chunks = self.retriever.retrieve(query=query, top_k=top_k, filters=filters)
        chunks = self.reranker.rerank(query, chunks)
        latency_ms = int((perf_counter() - start) * 1000)
        self._log_request(query=query, top_k=top_k, chunks=chunks, latency_ms=latency_ms)
        return chunks, latency_ms

    def _log_request(
        self,
        *,
        query: str,
        top_k: int,
        chunks: list[RetrievedChunk],
        latency_ms: int,
    ) -> None:
        self._ensure_retrieval_log_columns()
        log = RetrievalLog(
            query=query,
            retriever_name=self.retriever.retrieval_strategy,
            retrieved_chunk_ids=[chunk.chunk_id for chunk in chunks],
            scores={
                chunk.chunk_id: chunk.similarity_score if isfinite(chunk.similarity_score) else 0.0
                for chunk in chunks
            },
            top_k=top_k,
            latency_ms=latency_ms,
        )
        self.session.add(log)
        self.session.commit()

    def _ensure_retrieval_log_columns(self) -> None:
        bind = self.session.get_bind()
        if bind is not None and bind.dialect.name == "postgresql":
            self.session.execute(
                text("ALTER TABLE retrieval_logs ADD COLUMN IF NOT EXISTS top_k INTEGER")
            )


PgVectorRetriever = VectorRetriever

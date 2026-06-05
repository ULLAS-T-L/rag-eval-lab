from __future__ import annotations

from dataclasses import dataclass
import logging
from math import isfinite
import re
from time import perf_counter
from typing import Any, Optional, Protocol
from uuid import UUID

from sqlalchemy import Select, and_, distinct, func, or_, select, text
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document, Embedding, RetrievalLog
from app.retrieval.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalFilters:
    document_id: Optional[str] = None
    source_file: Optional[str] = None
    company: Optional[str] = None
    year: Optional[int] = None
    page: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_title: Optional[str] = None

    @classmethod
    def from_dict(cls, values: Optional[dict[str, Any]]) -> "RetrievalFilters":
        values = values or {}
        return cls(
            document_id=values.get("document_id"),
            source_file=values.get("source_file"),
            company=values.get("company"),
            year=values.get("year"),
            page=values.get("page"),
            page_start=values.get("page_start"),
            page_end=values.get("page_end"),
            section_title=values.get("section_title"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_file": self.source_file,
            "company": self.company,
            "year": self.year,
            "page": self.page,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section_title": self.section_title,
        }

    def without_section_title(self) -> "RetrievalFilters":
        return RetrievalFilters(
            document_id=self.document_id,
            source_file=self.source_file,
            company=self.company,
            year=self.year,
            page=self.page,
            page_start=self.page_start,
            page_end=self.page_end,
        )


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
        strategy: str = "metadata_then_vector",
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
        strategy: str = "metadata_then_vector",
    ) -> list[RetrievedChunk]:
        self._ensure_document_metadata_columns()
        query_vector = self.embedding_provider.embed_query(query)
        matched_document_ids = self._matched_document_ids(filters)
        candidate_chunk_count = self._candidate_chunk_count(filters)
        active_filters = filters
        if filters is not None and filters.section_title and candidate_chunk_count == 0:
            active_filters = filters.without_section_title()
            candidate_chunk_count = self._candidate_chunk_count(active_filters)
            matched_document_ids = self._matched_document_ids(active_filters)
        logger.debug(
            "retrieval_debug applied_filters=%s matched_document_ids=%s candidate_chunks=%s",
            active_filters.to_dict() if active_filters else {},
            matched_document_ids,
            candidate_chunk_count,
        )
        candidate_limit = max(top_k * 50, candidate_chunk_count or 0, 50)
        statement = self.build_query(query_vector=query_vector, top_k=candidate_limit, filters=active_filters)
        rows = self.session.execute(statement).all()
        chunks = [self._row_to_chunk(row) for row in rows]
        chunks = self._rank_chunks(query=query, chunks=chunks, filters=active_filters)[:top_k]
        logger.debug(
            "retrieval_debug retrieved_chunks=%s strategy=%s",
            len(chunks),
            strategy,
        )
        return chunks

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
            source_pattern = f"%{filters.source_file.lower()}%"
            predicates.append(
                or_(
                    func.lower(Document.source_file).like(source_pattern),
                    func.lower(Chunk.chunk_metadata["source_file"].as_string()).like(source_pattern),
                )
            )
        if filters.company:
            company_pattern = f"%{filters.company.lower()}%"
            predicates.append(
                or_(
                    func.lower(Document.company).like(company_pattern),
                    func.lower(Document.source_file).like(company_pattern),
                    func.lower(Chunk.chunk_metadata["source_file"].as_string()).like(company_pattern),
                )
            )
        if filters.year is not None:
            year_text = str(filters.year)
            predicates.append(
                (Document.doc_metadata["year"].as_string() == year_text)
                | (Document.source_uri.like(f"%{year_text}%"))
            )
        if filters.page is not None:
            predicates.append(Chunk.chunk_metadata["page_start"].as_integer() <= filters.page)
            predicates.append(Chunk.chunk_metadata["page_end"].as_integer() >= filters.page)
        if filters.page_start is not None:
            predicates.append(Chunk.chunk_metadata["page_end"].as_integer() >= filters.page_start)
        if filters.page_end is not None:
            predicates.append(Chunk.chunk_metadata["page_start"].as_integer() <= filters.page_end)
        if filters.section_title:
            section_pattern = f"%{filters.section_title.lower()}%"
            predicates.append(
                or_(
                    func.lower(Chunk.section_path).like(section_pattern),
                    func.lower(Chunk.section_title).like(section_pattern),
                    func.lower(Chunk.chunk_metadata["section_path"].as_string()).like(section_pattern),
                    func.lower(Chunk.chunk_metadata["section_title"].as_string()).like(section_pattern),
                )
            )
        return predicates

    def _matched_document_ids(self, filters: Optional[RetrievalFilters]) -> list[str]:
        predicates = self._filter_predicates(filters)
        statement = select(distinct(Document.id)).join(Chunk, Chunk.document_id == Document.id)
        if predicates:
            statement = statement.where(and_(*predicates))
        return [str(row[0]) for row in self.session.execute(statement).all()]

    def _candidate_chunk_count(self, filters: Optional[RetrievalFilters]) -> int:
        predicates = self._filter_predicates(filters)
        statement = (
            select(func.count(Chunk.id))
            .join(Document, Document.id == Chunk.document_id)
            .join(Embedding, Embedding.chunk_id == Chunk.id)
        )
        if predicates:
            statement = statement.where(and_(*predicates))
        return int(self.session.execute(statement).scalar() or 0)

    def _rank_chunks(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        filters: Optional[RetrievalFilters],
    ) -> list[RetrievedChunk]:
        query_terms = _terms(query)
        if not query_terms:
            return chunks

        intent_terms = _intent_terms(query_terms)
        focus_terms = _focus_terms(query_terms)
        ranked: list[tuple[float, RetrievedChunk]] = []
        for chunk in chunks:
            chunk_terms = _terms(chunk.chunk_text)
            keyword_terms = _metadata_terms(chunk.metadata.get("keywords", []))
            section_terms = _terms(
                " ".join(
                    str(value or "")
                    for value in (
                        chunk.metadata.get("section_title"),
                        chunk.metadata.get("section_path"),
                        chunk.citations[0].section_title if chunk.citations else "",
                        filters.section_title if filters else "",
                    )
                )
            )
            lexical_score = len(query_terms.intersection(chunk_terms)) / len(query_terms)
            focus_score = len(focus_terms.intersection(chunk_terms)) / max(len(focus_terms), 1)
            intent_score = len(intent_terms.intersection(chunk_terms)) / max(len(intent_terms), 1)
            keyword_score = len(query_terms.intersection(keyword_terms)) / len(query_terms)
            section_score = len(intent_terms.intersection(section_terms)) / max(len(intent_terms), 1)
            vector_score = max(0.0, min(1.0, chunk.similarity_score))
            combined_score = (
                0.10 * vector_score
                + 0.30 * lexical_score
                + 0.35 * focus_score
                + 0.15 * intent_score
                + 0.10 * keyword_score
                + 0.05 * section_score
            )
            if combined_score >= 0.08 or lexical_score >= 0.2 or focus_score > 0:
                ranked.append((combined_score, chunk))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                chunk_text=chunk.chunk_text,
                similarity_score=round(score, 4),
                document_metadata=chunk.document_metadata,
                citations=chunk.citations,
                metadata={**chunk.metadata, "vector_similarity_score": chunk.similarity_score},
            )
            for score, chunk in ranked
        ]

    def _ensure_document_metadata_columns(self) -> None:
        bind = self.session.get_bind()
        if bind is not None and bind.dialect.name == "postgresql":
            self.session.execute(
                text(
                    "ALTER TABLE documents "
                    "ADD COLUMN IF NOT EXISTS source_file VARCHAR(512), "
                    "ADD COLUMN IF NOT EXISTS company VARCHAR(255), "
                    "ADD COLUMN IF NOT EXISTS year INTEGER, "
                    "ADD COLUMN IF NOT EXISTS document_type VARCHAR(128)"
                )
            )
            self.session.execute(
                text(
                    "UPDATE documents SET "
                    "source_file = COALESCE(source_file, doc_metadata->>'source_file'), "
                    "company = COALESCE(company, CASE "
                    "WHEN lower(COALESCE(doc_metadata->>'source_file', source_uri)) LIKE '%apple%' THEN 'Apple' "
                    "WHEN lower(COALESCE(doc_metadata->>'source_file', source_uri)) LIKE '%microsoft%' THEN 'Microsoft' "
                    "WHEN lower(COALESCE(doc_metadata->>'source_file', source_uri)) LIKE '%amazon%' THEN 'Amazon' "
                    "ELSE NULL END), "
                    "document_type = COALESCE(document_type, CASE "
                    "WHEN lower(COALESCE(doc_metadata->>'source_file', source_uri)) LIKE '%10k%' "
                    "OR lower(COALESCE(doc_metadata->>'source_file', source_uri)) LIKE '%10-k%' THEN '10-K' "
                    "ELSE NULL END)"
                )
            )

    def _row_to_chunk(self, row: Any) -> RetrievedChunk:
        chunk: Chunk = row[0]
        document: Document = row[1]
        distance = float(row[2])
        similarity_score = 1.0 - distance if isfinite(distance) else 0.0
        metadata = dict(chunk.chunk_metadata or {})
        citation = Citation(
            document_id=str(document.id),
            source_file=document.source_file or metadata.get("source_file"),
            page_start=metadata.get("page_start"),
            page_end=metadata.get("page_end"),
            section_title=chunk.section_path,
            chunk_id=str(chunk.id),
        )
        return RetrievedChunk(
            chunk_id=str(chunk.id),
            chunk_text=chunk.content,
            similarity_score=similarity_score,
            document_metadata={
                **dict(document.doc_metadata or {}),
                "company": document.company,
                "year": document.year,
                "source_file": document.source_file,
                "document_type": document.document_type,
            },
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
        strategy: str = "metadata_then_vector",
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
        strategy: str = "metadata_then_vector",
    ) -> tuple[list[RetrievedChunk], int]:
        start = perf_counter()
        chunks = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            filters=filters,
            strategy=strategy,
        )
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


def _terms(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "around",
        "by",
        "describe",
        "disclose",
        "does",
        "from",
        "in",
        "of",
        "or",
        "the",
        "to",
        "what",
        "which",
    }
    terms = set()
    for raw_term in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower()):
        if raw_term in stopwords:
            continue
        terms.add(_normalize_term(raw_term))
    return terms


def _normalize_term(term: str) -> str:
    aliases = {
        "risks": "risk",
        "suppliers": "supplier",
        "manufacturers": "manufacturer",
        "disruptions": "disruption",
        "services": "service",
        "clouds": "cloud",
        "datacenters": "datacenter",
        "centers": "center",
    }
    if term in aliases:
        return aliases[term]
    if term.endswith("ies") and len(term) > 4:
        return f"{term[:-3]}y"
    if term.endswith("s") and len(term) > 4:
        return term[:-1]
    return term


def _intent_terms(query_terms: set[str]) -> set[str]:
    related = {
        "supply": {"supply", "supplier", "vendor", "component", "manufacturing", "manufacturer", "logistic"},
        "supplier": {"supply", "supplier", "vendor", "component", "manufacturer"},
        "chain": {"chain", "logistic", "distribution", "inventory", "transport"},
        "fulfillment": {"fulfillment", "distribution", "logistic", "inventory", "carrier", "transport"},
        "logistic": {"logistic", "distribution", "carrier", "transport", "supply"},
        "cloud": {"cloud", "datacenter", "server", "infrastructure", "capacity", "security", "energy"},
        "datacenter": {"datacenter", "data", "center", "infrastructure", "capacity", "energy"},
        "risk": {"risk", "disruption", "constraint", "dependence", "shortage", "failure", "availability"},
    }
    expanded = set(query_terms)
    for term in query_terms:
        expanded.update(related.get(term, set()))
    return expanded


def _focus_terms(query_terms: set[str]) -> set[str]:
    broad_terms = {"risk", "apple", "amazon", "microsoft", "describe", "disclose"}
    focus = {term for term in query_terms if term not in broad_terms}
    if "supply" in query_terms or "supplier" in query_terms:
        focus.update({"supply", "supplier", "component", "manufacturer", "logistic"})
    if "fulfillment" in query_terms or "logistic" in query_terms:
        focus.update({"fulfillment", "logistic", "distribution", "inventory", "carrier", "capacity"})
    if "cloud" in query_terms or "datacenter" in query_terms:
        focus.update({"cloud", "datacenter", "infrastructure", "capacity", "energy", "security"})
    return focus


def _metadata_terms(value: Any) -> set[str]:
    if isinstance(value, list):
        return _terms(" ".join(str(item) for item in value))
    if isinstance(value, str):
        return _terms(value)
    return set()

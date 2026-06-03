from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.agents.answer_generator import (
    GroundedAnswerGenerator,
    INSUFFICIENT_EVIDENCE_MESSAGE,
)
from app.agents.planner import QueryPlanner
from app.db.models import AnswerLog
from app.evaluation.trulens_tracing import TruLensTracer
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.rerankers import BaseReranker
from app.retrieval.retriever import RetrievalFilters, RetrievalService, VectorRetriever


@dataclass(frozen=True)
class AskResult:
    answer: str
    citations: list[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]
    applied_filters: dict[str, Any]
    retrieval_strategy: str
    confidence_score: None
    latency_ms: int
    token_usage: None
    trulens_run_id: str
    observability_scores: dict[str, float]


class ReasoningRouter:
    def __init__(
        self,
        session: Session,
        embedding_provider: EmbeddingProvider,
        reranker: BaseReranker,
        planner: Optional[QueryPlanner] = None,
        answer_generator: Optional[GroundedAnswerGenerator] = None,
        trulens_tracer: Optional[TruLensTracer] = None,
    ) -> None:
        self.session = session
        self.embedding_provider = embedding_provider
        self.reranker = reranker
        self.planner = planner or QueryPlanner()
        self.answer_generator = answer_generator or GroundedAnswerGenerator()
        self.trulens_tracer = trulens_tracer or TruLensTracer()

    def ask(self, query: str, top_k: int, filters: Optional[dict[str, Any]] = None) -> AskResult:
        started = perf_counter()
        plan = self.planner.plan(query=query, filters=filters)

        if plan.retrieval_strategy == "insufficient_query":
            latency_ms = int((perf_counter() - started) * 1000)
            trace = self.trulens_tracer.record_pipeline(
                question=query,
                retrieved_chunks=[],
                generated_answer=INSUFFICIENT_EVIDENCE_MESSAGE,
                citations=[],
                latency_ms=latency_ms,
                retrieval_latency_ms=0,
                generation_latency_ms=0,
            )
            result = AskResult(
                answer=INSUFFICIENT_EVIDENCE_MESSAGE,
                citations=[],
                retrieved_chunks=[],
                applied_filters=plan.filters,
                retrieval_strategy=plan.retrieval_strategy,
                confidence_score=None,
                latency_ms=latency_ms,
                token_usage=None,
                trulens_run_id=trace.run_id,
                observability_scores=trace.feedback_scores.to_dict(),
            )
            self._log_answer(query=query, result=result, trace_metadata=trace.to_metadata())
            return result

        retriever = VectorRetriever(session=self.session, embedding_provider=self.embedding_provider)
        retrieval_service = RetrievalService(
            session=self.session,
            retriever=retriever,
            reranker=self.reranker,
        )
        retrieval_filters = RetrievalFilters.from_dict(plan.filters)
        chunks, retrieval_latency_ms = retrieval_service.retrieve(
            query=plan.query,
            top_k=top_k,
            filters=retrieval_filters,
            strategy=plan.retrieval_strategy,
        )
        generated, generation_latency_ms = self.trulens_tracer.timed(
            lambda: self.answer_generator.generate(query=plan.query, chunks=chunks)
        )
        latency_ms = int((perf_counter() - started) * 1000)
        trace = self.trulens_tracer.record_pipeline(
            question=plan.query,
            retrieved_chunks=chunks,
            generated_answer=generated.answer,
            citations=generated.citations,
            latency_ms=latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
        )
        result = AskResult(
            answer=generated.answer,
            citations=generated.citations,
            retrieved_chunks=[self._chunk_payload(chunk) for chunk in chunks],
            applied_filters=plan.filters,
            retrieval_strategy=plan.retrieval_strategy,
            confidence_score=generated.confidence_score,
            latency_ms=latency_ms,
            token_usage=generated.token_usage,
            trulens_run_id=trace.run_id,
            observability_scores=trace.feedback_scores.to_dict(),
        )
        self._log_answer(query=query, result=result, trace_metadata=trace.to_metadata())
        return result

    def _chunk_payload(self, chunk) -> dict[str, Any]:
        citation = chunk.citations[0] if chunk.citations else None
        return {
            "chunk_id": chunk.chunk_id,
            "source_file": citation.source_file if citation else chunk.metadata.get("source_file"),
            "page_start": citation.page_start if citation else chunk.metadata.get("page_start"),
            "page_end": citation.page_end if citation else chunk.metadata.get("page_end"),
            "section_title": citation.section_title if citation else chunk.metadata.get("section_title"),
            "similarity_score": chunk.similarity_score,
            "metadata": chunk.metadata,
            "text_preview": chunk.chunk_text[:500],
        }

    def _log_answer(
        self,
        query: str,
        result: AskResult,
        trace_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self._ensure_answer_log_columns()
        log = AnswerLog(
            trulens_run_id=result.trulens_run_id,
            question=query,
            answer=result.answer,
            model="grounded-deterministic",
            retrieved_chunk_ids=[chunk["chunk_id"] for chunk in result.retrieved_chunks],
            applied_filters=result.applied_filters,
            retrieval_strategy=result.retrieval_strategy,
            citations=result.citations,
            latency_ms=result.latency_ms,
            answer_metadata={
                "query": query,
                "token_usage": result.token_usage,
                "confidence_score": result.confidence_score,
                "trulens": trace_metadata or {},
                "observability_scores": result.observability_scores,
            },
        )
        self.session.add(log)
        self.session.commit()

    def _ensure_answer_log_columns(self) -> None:
        bind = self.session.get_bind()
        if bind is not None and bind.dialect.name == "postgresql":
            self.session.execute(
                text(
                    "ALTER TABLE answer_logs "
                    "ADD COLUMN IF NOT EXISTS retrieved_chunk_ids JSON DEFAULT '[]', "
                    "ADD COLUMN IF NOT EXISTS applied_filters JSON DEFAULT '{}', "
                    "ADD COLUMN IF NOT EXISTS retrieval_strategy VARCHAR(255), "
                    "ADD COLUMN IF NOT EXISTS latency_ms INTEGER, "
                    "ADD COLUMN IF NOT EXISTS trulens_run_id VARCHAR(64)"
                )
            )

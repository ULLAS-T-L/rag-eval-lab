from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
import re
from typing import Any, Optional, TypeVar
from uuid import uuid4

T = TypeVar("T")


@dataclass(frozen=True)
class TruLensFeedbackScores:
    context_relevance: float
    groundedness: float
    answer_relevance: float

    def to_dict(self) -> dict[str, float]:
        return {
            "context_relevance": self.context_relevance,
            "groundedness": self.groundedness,
            "answer_relevance": self.answer_relevance,
        }


@dataclass(frozen=True)
class TruLensTraceRecord:
    run_id: str
    question: str
    retrieved_chunks: list[dict[str, Any]]
    similarity_scores: dict[str, float]
    generated_answer: str
    citations: list[dict[str, Any]]
    latency_ms: int
    retrieval_latency_ms: int
    generation_latency_ms: int
    feedback_scores: TruLensFeedbackScores

    def to_metadata(self) -> dict[str, Any]:
        return {
            "trulens_run_id": self.run_id,
            "question": self.question,
            "retrieved_chunks": self.retrieved_chunks,
            "similarity_scores": self.similarity_scores,
            "generated_answer": self.generated_answer,
            "citations": self.citations,
            "latency_ms": self.latency_ms,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "generation_latency_ms": self.generation_latency_ms,
            "feedback_scores": self.feedback_scores.to_dict(),
        }


class TruLensTracer:
    """Provider-agnostic tracing and feedback scoring for the online RAG path.

    The class intentionally avoids importing TruLens at request time. That keeps the
    application usable in environments where a dashboard dependency is unavailable while
    still producing TruLens-style trace records and feedback scores.
    """

    def trace(self, name: str, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        return fn(*args, **kwargs)

    def timed(self, fn: Callable[[], T]) -> tuple[T, int]:
        started = perf_counter()
        result = fn()
        return result, int((perf_counter() - started) * 1000)

    def record_pipeline(
        self,
        *,
        question: str,
        retrieved_chunks: Sequence[Any],
        generated_answer: str,
        citations: Sequence[dict[str, Any]],
        latency_ms: int,
        retrieval_latency_ms: int,
        generation_latency_ms: int,
        run_id: Optional[str] = None,
    ) -> TruLensTraceRecord:
        chunk_payloads = [self._chunk_payload(chunk) for chunk in retrieved_chunks]
        similarity_scores = {
            str(chunk["chunk_id"]): float(chunk["similarity_score"])
            for chunk in chunk_payloads
            if chunk.get("chunk_id") is not None
        }
        context_texts = [str(chunk.get("text", "")) for chunk in chunk_payloads]
        feedback_scores = TruLensFeedbackScores(
            context_relevance=self.context_relevance(question, context_texts),
            groundedness=self.groundedness(generated_answer, context_texts),
            answer_relevance=self.answer_relevance(question, generated_answer),
        )
        return TruLensTraceRecord(
            run_id=run_id or uuid4().hex,
            question=question,
            retrieved_chunks=chunk_payloads,
            similarity_scores=similarity_scores,
            generated_answer=generated_answer,
            citations=list(citations),
            latency_ms=latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            feedback_scores=feedback_scores,
        )

    def context_relevance(self, question: str, contexts: Sequence[str]) -> float:
        question_terms = _terms(question)
        if not question_terms or not contexts:
            return 0.0
        scores = []
        for context in contexts:
            context_terms = _terms(context)
            scores.append(len(question_terms.intersection(context_terms)) / len(question_terms))
        return round(sum(scores) / len(scores), 4)

    def groundedness(self, answer: str, contexts: Sequence[str]) -> float:
        answer_terms = _terms(answer)
        if not answer_terms:
            return 0.0
        context_terms = _terms(" ".join(contexts))
        return round(len(answer_terms.intersection(context_terms)) / len(answer_terms), 4)

    def answer_relevance(self, question: str, answer: str) -> float:
        question_terms = _terms(question)
        answer_terms = _terms(answer)
        if not question_terms or not answer_terms:
            return 0.0
        precision = len(question_terms.intersection(answer_terms)) / len(answer_terms)
        recall = len(question_terms.intersection(answer_terms)) / len(question_terms)
        if precision + recall == 0:
            return 0.0
        return round(2 * precision * recall / (precision + recall), 4)

    def _chunk_payload(self, chunk: Any) -> dict[str, Any]:
        if isinstance(chunk, dict):
            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text") or chunk.get("chunk_text") or chunk.get("text_preview") or ""
            similarity_score = chunk.get("similarity_score", 0.0)
            citations = chunk.get("citations", [])
            metadata = chunk.get("metadata", {})
        else:
            chunk_id = getattr(chunk, "chunk_id", None)
            text = getattr(chunk, "chunk_text", "")
            similarity_score = getattr(chunk, "similarity_score", 0.0)
            citations = [
                citation.__dict__
                for citation in getattr(chunk, "citations", [])
                if hasattr(citation, "__dict__")
            ]
            metadata = getattr(chunk, "metadata", {})
        return {
            "chunk_id": str(chunk_id) if chunk_id is not None else None,
            "text": str(text),
            "similarity_score": float(similarity_score or 0.0),
            "citations": citations,
            "metadata": metadata or {},
        }


def _terms(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "by",
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
    return {
        term
        for term in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())
        if term not in stopwords
    }

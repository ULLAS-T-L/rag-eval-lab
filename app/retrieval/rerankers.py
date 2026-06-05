from __future__ import annotations

import re
from typing import Protocol

from app.retrieval.retriever import RetrievedChunk


class BaseReranker(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        ...


class NoOpReranker:
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return chunks


class LexicalReranker:
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        query_terms = _terms(query)
        if not query_terms:
            return chunks
        return sorted(
            chunks,
            key=lambda chunk: (
                len(query_terms.intersection(_terms(chunk.chunk_text))) / len(query_terms),
                chunk.similarity_score,
            ),
            reverse=True,
        )


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "CrossEncoderReranker requires `sentence-transformers`. Install requirements.txt."
            ) from exc
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [(query, chunk.chunk_text[:2000]) for chunk in chunks]
        scores = [float(score) for score in self._model.predict(pairs)]
        scored = zip(scores, chunks)
        return [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                chunk_text=chunk.chunk_text,
                similarity_score=round(score, 4),
                document_metadata=chunk.document_metadata,
                citations=chunk.citations,
                metadata={**chunk.metadata, "pre_rerank_similarity_score": chunk.similarity_score},
            )
            for score, chunk in sorted(scored, key=lambda item: item[0], reverse=True)
        ]


def _terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())
        if term
        not in {
            "and",
            "are",
            "from",
            "the",
            "to",
            "what",
            "which",
        }
    }

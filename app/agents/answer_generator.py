from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from app.retrieval.retriever import RetrievedChunk


INSUFFICIENT_EVIDENCE_MESSAGE = "I do not have enough evidence from the retrieved documents."


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    citations: list[dict[str, Any]]
    confidence_score: None = None
    token_usage: None = None


class GroundedAnswerGenerator:
    """Deterministic grounded generator that only uses retrieved chunk text."""

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
        evidence = self._select_evidence(query, chunks)
        if not evidence:
            return GeneratedAnswer(answer=INSUFFICIENT_EVIDENCE_MESSAGE, citations=[])

        answer_parts: list[str] = []
        citations: list[dict[str, Any]] = []
        seen_citations: set[str] = set()

        for chunk, sentence in evidence:
            citation = chunk.citations[0] if chunk.citations else None
            if citation is None:
                continue
            answer_parts.append(f"{sentence.strip()} [{len(citations) + 1}]")
            if citation.chunk_id not in seen_citations:
                citations.append(
                    {
                        "source_file": citation.source_file,
                        "page_start": citation.page_start,
                        "page_end": citation.page_end,
                        "section_title": citation.section_title,
                        "chunk_id": citation.chunk_id,
                    }
                )
                seen_citations.add(citation.chunk_id)

        if not answer_parts or not citations:
            return GeneratedAnswer(answer=INSUFFICIENT_EVIDENCE_MESSAGE, citations=[])

        return GeneratedAnswer(answer=" ".join(answer_parts), citations=citations)

    def _select_evidence(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[tuple[RetrievedChunk, str]]:
        query_terms = self._terms(query)
        evidence: list[tuple[RetrievedChunk, str]] = []

        for chunk in chunks:
            best_sentence = self._best_sentence(chunk.chunk_text, query_terms)
            lexical_overlap = len(query_terms.intersection(self._terms(best_sentence)))
            strong_vector_score = chunk.similarity_score >= 0.05
            if best_sentence and (lexical_overlap > 0 or strong_vector_score):
                evidence.append((chunk, best_sentence))
            if len(evidence) >= 3:
                break

        return evidence

    def _best_sentence(self, text: str, query_terms: set[str]) -> str:
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
        if not sentences:
            return text[:500].strip()
        return max(sentences, key=lambda sentence: len(query_terms.intersection(self._terms(sentence))))[:500]

    def _terms(self, text: str) -> set[str]:
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "did",
            "from",
            "in",
            "of",
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

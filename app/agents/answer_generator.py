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
            answer_parts.append(f"- {sentence.strip()} [{len(citations)}]")

        if not answer_parts or not citations:
            return GeneratedAnswer(answer=INSUFFICIENT_EVIDENCE_MESSAGE, citations=[])

        return GeneratedAnswer(
            answer=f"Answer to the question: {query}\n" + "\n".join(answer_parts),
            citations=citations,
        )

    def _select_evidence(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[tuple[RetrievedChunk, str]]:
        query_terms = self._terms(query)
        intent_terms = self._intent_terms(query_terms)
        evidence: list[tuple[RetrievedChunk, str]] = []

        for chunk in chunks:
            best_sentence = self._best_sentence(chunk.chunk_text, query_terms, intent_terms)
            sentence_terms = self._terms(best_sentence)
            lexical_overlap = len(query_terms.intersection(sentence_terms))
            intent_overlap = len(intent_terms.intersection(sentence_terms))
            if best_sentence and lexical_overlap >= 1 and intent_overlap >= 2:
                evidence.append((chunk, best_sentence))
            if len(evidence) >= 3:
                break

        return evidence

    def _best_sentence(self, text: str, query_terms: set[str], intent_terms: set[str]) -> str:
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
        if not sentences:
            return text[:500].strip()
        return max(
            sentences,
            key=lambda sentence: (
                len(intent_terms.intersection(self._terms(sentence))),
                len(query_terms.intersection(self._terms(sentence))),
            ),
        )[:500]

    def _terms(self, text: str) -> set[str]:
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "around",
            "describe",
            "disclose",
            "did",
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
            self._normalize_term(term)
            for term in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())
            if term not in stopwords
        }

    def _normalize_term(self, term: str) -> str:
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

    def _intent_terms(self, query_terms: set[str]) -> set[str]:
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

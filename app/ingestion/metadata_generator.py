from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Any, Protocol

from app.ingestion.question_generator import DeterministicQuestionGenerator, QuestionGenerator


STOPWORDS = {
    "about",
    "after",
    "and",
    "are",
    "because",
    "been",
    "before",
    "being",
    "between",
    "can",
    "could",
    "from",
    "into",
    "more",
    "most",
    "other",
    "our",
    "that",
    "the",
    "their",
    "there",
    "this",
    "those",
    "through",
    "under",
    "were",
    "what",
    "when",
    "where",
    "which",
    "with",
}


class MetadataGenerator(Protocol):
    def generate_document_metadata(self, source_file: str, source_path: str | None = None) -> dict[str, Any]:
        ...

    def generate_chunk_metadata(
        self,
        text: str,
        *,
        company: str | None = None,
        section_title: str | None = None,
        chunk_type: str = "text",
        max_questions: int = 3,
    ) -> dict[str, Any]:
        ...


class DeterministicMetadataGenerator:
    def __init__(self, question_generator: QuestionGenerator | None = None) -> None:
        self.question_generator = question_generator or DeterministicQuestionGenerator()

    def generate_document_metadata(self, source_file: str, source_path: str | None = None) -> dict[str, Any]:
        inferred = infer_document_metadata_from_filename(source_file)
        inferred["source_file"] = source_file
        if source_path:
            inferred["source_path"] = source_path
        return inferred

    def generate_chunk_metadata(
        self,
        text: str,
        *,
        company: str | None = None,
        section_title: str | None = None,
        chunk_type: str = "text",
        max_questions: int = 3,
    ) -> dict[str, Any]:
        keywords = extract_keywords(text)
        entities = extract_entities(text)
        summary = summarize_text(text)
        synthetic_questions = self.question_generator.generate(
            text,
            company=company,
            section_title=section_title,
            keywords=keywords,
            max_questions=max_questions,
        )
        return {
            "summary": summary,
            "keywords": keywords,
            "entities": entities,
            "synthetic_questions": synthetic_questions,
            "chunk_type": chunk_type,
        }


def infer_document_metadata_from_filename(source_file: str) -> dict[str, Any]:
    filename = source_file.lower()
    company_map = {
        "apple": "Apple",
        "microsoft": "Microsoft",
        "amazon": "Amazon",
        "kirkland": "Kirkland",
    }
    company = next((value for key, value in company_map.items() if key in filename), None)
    year_match = re.search(r"\b(20\d{2})\b", filename)
    document_type = "10-K" if "10k" in filename.replace("-", "").replace("_", "") else None
    return {
        "company": company,
        "year": int(year_match.group(1)) if year_match else None,
        "document_type": document_type,
    }


def summarize_text(text: str, max_chars: int = 240) -> str:
    stripped = " ".join(text.split())
    if not stripped:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+", stripped)[0]
    return sentence[:max_chars].strip()


def extract_keywords(text: str, limit: int = 6) -> list[str]:
    words = [
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text)
        if word.lower() not in STOPWORDS
    ]
    counts = Counter(words)
    return [word for word, _ in counts.most_common(limit)]


def extract_entities(text: str, limit: int = 5) -> list[str]:
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9&.-]{2,}\b", text)
    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped[:limit]

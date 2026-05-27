from __future__ import annotations

from typing import Protocol


class QuestionGenerator(Protocol):
    def generate(
        self,
        text: str,
        *,
        company: str | None = None,
        section_title: str | None = None,
        keywords: list[str] | None = None,
        max_questions: int = 3,
    ) -> list[str]:
        ...


class DeterministicQuestionGenerator:
    def generate(
        self,
        text: str,
        *,
        company: str | None = None,
        section_title: str | None = None,
        keywords: list[str] | None = None,
        max_questions: int = 3,
    ) -> list[str]:
        keywords = [keyword for keyword in (keywords or []) if keyword]
        focus = keywords[:2]
        company_prefix = f"{company} " if company else ""
        section_phrase = section_title or "this section"

        questions = [
            f"What does {company_prefix}{section_phrase} say about {focus[0]}?" if focus else f"What does {company_prefix}{section_phrase} say?",
            f"Which risks or obligations are described in {section_phrase}?" if section_title else "Which risks or obligations are described here?",
            f"What evidence in this chunk supports the main point?" if focus else "What evidence in this chunk supports the main point?",
        ]

        return questions[: max(2, min(max_questions, len(questions)))]

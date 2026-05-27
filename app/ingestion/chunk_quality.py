from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkQualityReport:
    score: float
    issues: list[str]


class ChunkQualityEvaluator:
    def evaluate(
        self,
        text: str,
        *,
        chunk_type: str,
        token_count: int,
        section_title: str | None = None,
    ) -> ChunkQualityReport:
        issues: list[str] = []
        score = 1.0

        if token_count < 20:
            issues.append("short_chunk")
            score -= 0.2
        if not section_title:
            issues.append("missing_section_title")
            score -= 0.15
        if chunk_type == "table":
            score += 0.05
        if len(text.strip()) == 0:
            issues.append("empty_chunk")
            score = 0.0

        return ChunkQualityReport(score=max(0.0, min(score, 1.0)), issues=issues)

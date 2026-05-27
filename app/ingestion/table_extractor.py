from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TableDetection:
    is_table: bool
    confidence: float


class TableExtractor:
    """Heuristic table detector that preserves table-like blocks as single chunks."""

    def is_table_like(self, text: str) -> bool:
        return self.detect(text).is_table

    def detect(self, text: str) -> TableDetection:
        normalized = text.strip()
        if not normalized:
            return TableDetection(is_table=False, confidence=0.0)

        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        if len(lines) < 2:
            return TableDetection(is_table=False, confidence=0.0)

        score = 0.0
        if "|" in normalized or "\t" in normalized:
            score += 0.45
        if any(re.search(r"\s{2,}", line) for line in lines):
            score += 0.25
        if sum(1 for line in lines if re.search(r"\d", line)) >= max(2, len(lines) // 2):
            score += 0.15
        if sum(1 for line in lines if len(line.split()) >= 4) >= 2:
            score += 0.15

        return TableDetection(is_table=score >= 0.45, confidence=min(score, 1.0))

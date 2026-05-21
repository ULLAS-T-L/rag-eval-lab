from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RagasEvaluationResult:
    metric_name: str
    score: float | None
    details: dict


class RagasEvaluator:
    def evaluate(self, question: str, answer: str, contexts: list[str]) -> list[RagasEvaluationResult]:
        return [
            RagasEvaluationResult(
                metric_name="placeholder",
                score=None,
                details={"reason": "RAGAS metrics not configured yet"},
            )
        ]

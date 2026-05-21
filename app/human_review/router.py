from __future__ import annotations

def requires_human_review(evaluation: dict, threshold: float = 0.7) -> bool:
    score = evaluation.get("confidence")
    return score is None or float(score) < threshold

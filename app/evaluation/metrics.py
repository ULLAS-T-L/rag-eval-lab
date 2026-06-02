from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
import re
from typing import Any, Iterable, Sequence

from app.evaluation.datasets import EvaluationExample


METRIC_COLUMNS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


@dataclass(frozen=True)
class EvaluationScoreRow:
    run_id: str
    question: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    details: dict[str, Any]

    def to_csv_row(self, timestamp: str) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "question": self.question,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "timestamp": timestamp,
        }


@dataclass(frozen=True)
class AggregateReport:
    averages: dict[str, float]
    worst_questions: list[dict[str, Any]]
    best_questions: list[dict[str, Any]]


def aggregate_scores(
    rows: Sequence[EvaluationScoreRow],
    *,
    question_count: int = 5,
) -> AggregateReport:
    if not rows:
        return AggregateReport(averages={}, worst_questions=[], best_questions=[])

    averages = {
        metric: round(mean(getattr(row, metric) for row in rows), 4)
        for metric in METRIC_COLUMNS
    }
    ranked = sorted(rows, key=lambda row: _overall_score(row))
    return AggregateReport(
        averages=averages,
        worst_questions=[_question_summary(row) for row in ranked[:question_count]],
        best_questions=[_question_summary(row) for row in reversed(ranked[-question_count:])],
    )


def lexical_fallback_scores(
    examples: Sequence[EvaluationExample],
    *,
    run_id: str,
) -> list[EvaluationScoreRow]:
    rows: list[EvaluationScoreRow] = []
    for example in examples:
        context_text = " ".join(example.contexts)
        faithfulness = _coverage(example.answer, context_text)
        answer_relevancy = _f1(_terms(example.question), _terms(example.answer))
        context_precision = _context_precision(example.question, example.contexts)
        context_recall = _coverage(example.ground_truth, context_text)
        rows.append(
            EvaluationScoreRow(
                run_id=run_id,
                question=example.question,
                faithfulness=faithfulness,
                answer_relevancy=answer_relevancy,
                context_precision=context_precision,
                context_recall=context_recall,
                details={
                    "evaluator": "lexical_fallback",
                    "reason": "RAGAS was unavailable or no provider adapters were supplied.",
                },
            )
        )
    return rows


def rows_from_ragas_result(
    result: Any,
    examples: Sequence[EvaluationExample],
    *,
    run_id: str,
) -> list[EvaluationScoreRow]:
    dataframe = result.to_pandas() if hasattr(result, "to_pandas") else result
    records = dataframe.to_dict("records") if hasattr(dataframe, "to_dict") else list(dataframe)
    rows: list[EvaluationScoreRow] = []
    for index, record in enumerate(records):
        example = examples[index]
        rows.append(
            EvaluationScoreRow(
                run_id=run_id,
                question=example.question,
                faithfulness=_coerce_score(record.get("faithfulness")),
                answer_relevancy=_coerce_score(
                    record.get("answer_relevancy", record.get("answer_relevance"))
                ),
                context_precision=_coerce_score(record.get("context_precision")),
                context_recall=_coerce_score(record.get("context_recall")),
                details={"evaluator": "ragas"},
            )
        )
    return rows


def _question_summary(row: EvaluationScoreRow) -> dict[str, Any]:
    return {
        "question": row.question,
        "overall": round(_overall_score(row), 4),
        **{metric: round(getattr(row, metric), 4) for metric in METRIC_COLUMNS},
    }


def _overall_score(row: EvaluationScoreRow) -> float:
    return mean(getattr(row, metric) for metric in METRIC_COLUMNS)


def _context_precision(question: str, contexts: Sequence[str]) -> float:
    query_terms = _terms(question)
    if not contexts:
        return 0.0
    relevant = [context for context in contexts if _terms(context).intersection(query_terms)]
    return round(len(relevant) / len(contexts), 4)


def _coverage(source: str, target: str) -> float:
    source_terms = _terms(source)
    if not source_terms:
        return 0.0
    return round(len(source_terms.intersection(_terms(target))) / len(source_terms), 4)


def _f1(expected: set[str], observed: set[str]) -> float:
    if not expected or not observed:
        return 0.0
    precision = len(expected.intersection(observed)) / len(observed)
    recall = len(expected.intersection(observed)) / len(expected)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


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
    }
    return {
        term
        for term in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())
        if term not in stopwords
    }


def _coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(1.0, score)), 4)

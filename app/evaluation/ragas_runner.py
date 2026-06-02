from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from typing import Any, Optional, Sequence
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.models import EvaluationScore
from app.evaluation.datasets import EvaluationExample, to_ragas_dataset
from app.evaluation.metrics import (
    AggregateReport,
    EvaluationScoreRow,
    aggregate_scores,
    lexical_fallback_scores,
    rows_from_ragas_result,
)


REPORT_PATH = Path("evals/reports/ragas_report.csv")
SUMMARY_PATH = Path("evals/reports/ragas_summary.json")


@dataclass(frozen=True)
class RagasRunResult:
    run_id: str
    rows: list[EvaluationScoreRow]
    aggregate_report: AggregateReport
    report_path: Path
    summary_path: Path


class RagasRunner:
    def __init__(
        self,
        *,
        session: Optional[Session] = None,
        llm: Any = None,
        embeddings: Any = None,
        metrics: Optional[Sequence[Any]] = None,
        allow_fallback: bool = True,
    ) -> None:
        self.session = session
        self.llm = llm
        self.embeddings = embeddings
        self.metrics = metrics
        self.allow_fallback = allow_fallback

    def run(
        self,
        examples: Sequence[EvaluationExample],
        *,
        run_id: Optional[str] = None,
        report_path: Path = REPORT_PATH,
        summary_path: Path = SUMMARY_PATH,
    ) -> RagasRunResult:
        active_run_id = run_id or uuid4().hex
        rows = self._evaluate(examples=examples, run_id=active_run_id)
        aggregate_report = aggregate_scores(rows)
        timestamp = datetime.now(timezone.utc).isoformat()

        if self.session is not None:
            self.persist(rows=rows, timestamp=timestamp)
        self.export_csv(rows=rows, path=report_path, timestamp=timestamp)
        self.export_summary(report=aggregate_report, path=summary_path)

        return RagasRunResult(
            run_id=active_run_id,
            rows=rows,
            aggregate_report=aggregate_report,
            report_path=report_path,
            summary_path=summary_path,
        )

    def _evaluate(
        self,
        *,
        examples: Sequence[EvaluationExample],
        run_id: str,
    ) -> list[EvaluationScoreRow]:
        if not examples:
            return []

        try:
            result = self._run_ragas(examples)
            return rows_from_ragas_result(result, examples, run_id=run_id)
        except Exception as exc:
            if not self.allow_fallback:
                raise
            rows = lexical_fallback_scores(examples, run_id=run_id)
            return [
                EvaluationScoreRow(
                    run_id=row.run_id,
                    question=row.question,
                    faithfulness=row.faithfulness,
                    answer_relevancy=row.answer_relevancy,
                    context_precision=row.context_precision,
                    context_recall=row.context_recall,
                    details={**row.details, "ragas_error": str(exc)},
                )
                for row in rows
            ]

    def _run_ragas(self, examples: Sequence[EvaluationExample]) -> Any:
        from datasets import Dataset
        from ragas import evaluate

        dataset = Dataset.from_dict(to_ragas_dataset(examples))
        metric_objects = list(self.metrics) if self.metrics is not None else self._default_ragas_metrics()
        kwargs: dict[str, Any] = {"metrics": metric_objects}
        if self.llm is not None:
            kwargs["llm"] = self.llm
        if self.embeddings is not None:
            kwargs["embeddings"] = self.embeddings
        return evaluate(dataset, **kwargs)

    def _default_ragas_metrics(self) -> list[Any]:
        try:
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

            return [faithfulness, answer_relevancy, context_precision, context_recall]
        except ImportError:
            from ragas.metrics import (
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
                Faithfulness,
            )

            return [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]

    def persist(self, *, rows: Sequence[EvaluationScoreRow], timestamp: str) -> None:
        if self.session is None:
            return
        self._ensure_evaluation_score_columns()
        for row in rows:
            self.session.add(
                EvaluationScore(
                    run_id=row.run_id,
                    question=row.question,
                    faithfulness=row.faithfulness,
                    answer_relevancy=row.answer_relevancy,
                    context_precision=row.context_precision,
                    context_recall=row.context_recall,
                    details=row.details,
                    timestamp=datetime.fromisoformat(timestamp),
                )
            )
        self.session.commit()

    def export_csv(
        self,
        *,
        rows: Sequence[EvaluationScoreRow],
        path: Path,
        timestamp: str,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "run_id",
            "question",
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "timestamp",
        ]
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(row.to_csv_row(timestamp) for row in rows)

    def export_summary(self, *, report: AggregateReport, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "average_scores": report.averages,
            "worst_questions": report.worst_questions,
            "best_questions": report.best_questions,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _ensure_evaluation_score_columns(self) -> None:
        bind = self.session.get_bind()
        if bind is None or bind.dialect.name != "postgresql":
            return
        self.session.execute(
            text(
                "CREATE TABLE IF NOT EXISTS evaluation_scores ("
                "id UUID PRIMARY KEY, "
                "run_id VARCHAR(64) NOT NULL, "
                "question TEXT NOT NULL, "
                "faithfulness DOUBLE PRECISION, "
                "answer_relevancy DOUBLE PRECISION, "
                "context_precision DOUBLE PRECISION, "
                "context_recall DOUBLE PRECISION, "
                "timestamp TIMESTAMPTZ DEFAULT now(), "
                "details JSON DEFAULT '{}', "
                "created_at TIMESTAMPTZ DEFAULT now(), "
                "updated_at TIMESTAMPTZ DEFAULT now()"
                ")"
            )
        )
        self.session.execute(
            text(
                "ALTER TABLE evaluation_scores "
                "ADD COLUMN IF NOT EXISTS run_id VARCHAR(64), "
                "ADD COLUMN IF NOT EXISTS question TEXT, "
                "ADD COLUMN IF NOT EXISTS faithfulness DOUBLE PRECISION, "
                "ADD COLUMN IF NOT EXISTS answer_relevancy DOUBLE PRECISION, "
                "ADD COLUMN IF NOT EXISTS context_precision DOUBLE PRECISION, "
                "ADD COLUMN IF NOT EXISTS context_recall DOUBLE PRECISION, "
                "ADD COLUMN IF NOT EXISTS timestamp TIMESTAMPTZ DEFAULT now(), "
                "ADD COLUMN IF NOT EXISTS details JSON DEFAULT '{}'"
            )
        )
        self.session.execute(
            text(
                "DO $$ BEGIN "
                "IF EXISTS (SELECT 1 FROM information_schema.columns "
                "WHERE table_name = 'evaluation_scores' AND column_name = 'evaluator') THEN "
                "ALTER TABLE evaluation_scores ALTER COLUMN evaluator DROP NOT NULL; "
                "END IF; "
                "IF EXISTS (SELECT 1 FROM information_schema.columns "
                "WHERE table_name = 'evaluation_scores' AND column_name = 'metric_name') THEN "
                "ALTER TABLE evaluation_scores ALTER COLUMN metric_name DROP NOT NULL; "
                "END IF; "
                "END $$;"
            )
        )

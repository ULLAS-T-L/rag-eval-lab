from __future__ import annotations

from datetime import datetime

from app.db.models import EvaluationScore
from app.evaluation.datasets import (
    EvaluationDatasetBuilder,
    MANUAL_BENCHMARK_QUESTIONS,
    to_ragas_dataset,
)
from app.evaluation.metrics import EvaluationScoreRow, aggregate_scores
from app.evaluation.ragas_runner import RagasRunner


def test_manual_benchmark_dataset_uses_required_schema() -> None:
    def answer_fn(question, filters):
        return f"Answer for {filters['company']}", [f"Context for {filters['company']}"]

    examples = EvaluationDatasetBuilder(answer_fn=answer_fn).from_benchmark_questions()
    dataset = to_ragas_dataset(examples)

    assert set(dataset) == {"question", "ground_truth", "contexts", "answer"}
    assert len(examples) == 3
    assert {example.metadata["filters"]["company"] for example in examples} == {
        "Apple",
        "Amazon",
        "Microsoft",
    }


def test_manual_benchmarks_cover_required_companies() -> None:
    companies = {benchmark.filters["company"] for benchmark in MANUAL_BENCHMARK_QUESTIONS}

    assert companies == {"Apple", "Amazon", "Microsoft"}


def test_aggregate_report_contains_average_best_and_worst_questions() -> None:
    rows = [
        EvaluationScoreRow("run-1", "bad", 0.1, 0.2, 0.3, 0.4, {}),
        EvaluationScoreRow("run-1", "good", 0.9, 0.8, 0.7, 0.6, {}),
    ]

    report = aggregate_scores(rows, question_count=1)

    assert report.averages["faithfulness"] == 0.5
    assert report.worst_questions[0]["question"] == "bad"
    assert report.best_questions[0]["question"] == "good"


def test_runner_exports_csv_and_summary_with_fallback(tmp_path) -> None:
    examples = EvaluationDatasetBuilder(
        answer_fn=lambda question, filters: (
            "Apple supplier availability affects operations.",
            ["Apple relies on suppliers for availability."],
        )
    ).from_benchmark_questions(MANUAL_BENCHMARK_QUESTIONS[:1])

    runner = RagasRunner(allow_fallback=True)
    runner._run_ragas = lambda examples: (_ for _ in ()).throw(RuntimeError("no provider"))

    result = runner.run(
        examples,
        run_id="run-1",
        report_path=tmp_path / "ragas_report.csv",
        summary_path=tmp_path / "ragas_summary.json",
    )

    assert result.report_path.exists()
    assert result.summary_path.exists()
    csv_text = result.report_path.read_text(encoding="utf-8")
    assert "run_id,question,faithfulness,answer_relevancy,context_precision,context_recall,timestamp" in csv_text
    assert "run-1" in csv_text
    assert result.aggregate_report.averages


def test_runner_persists_one_evaluation_score_per_question() -> None:
    class FakeSession:
        def __init__(self) -> None:
            self.logged = []
            self.commits = 0

        def add(self, item) -> None:
            self.logged.append(item)

        def commit(self) -> None:
            self.commits += 1

        def get_bind(self):
            return None

    session = FakeSession()
    row = EvaluationScoreRow("run-1", "question", 1.0, 0.9, 0.8, 0.7, {"evaluator": "test"})

    RagasRunner(session=session).persist(rows=[row], timestamp=datetime.utcnow().isoformat())

    assert session.commits == 1
    assert len(session.logged) == 1
    assert isinstance(session.logged[0], EvaluationScore)
    assert session.logged[0].run_id == "run-1"
    assert session.logged[0].faithfulness == 1.0

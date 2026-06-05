from __future__ import annotations

from datetime import datetime
import json

import pytest

from app.db.models import EvaluationScore
from app.evaluation.datasets import (
    EvaluationExample,
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
        debug_dataset_path=tmp_path / "ragas_debug_dataset.json",
    )

    assert result.report_path.exists()
    assert result.summary_path.exists()
    debug_path = tmp_path / "ragas_debug_dataset.json"
    assert debug_path.exists()
    debug_payload = json.loads(debug_path.read_text(encoding="utf-8"))
    assert debug_payload[0]["contexts"] == ["Apple relies on suppliers for availability."]
    assert debug_payload[0]["context_count"] == 1
    csv_text = result.report_path.read_text(encoding="utf-8")
    assert "run_id,question,faithfulness,answer_relevancy,context_precision,context_recall,timestamp" in csv_text
    assert "run-1" in csv_text
    assert result.aggregate_report.averages


def test_runner_fails_when_contexts_are_empty(tmp_path) -> None:
    examples = [
        EvaluationExample(
            question="What supply chain or supplier risks does Apple disclose?",
            ground_truth="Apple discloses supplier and supply chain risks.",
            contexts=[],
            answer="Apple discloses supplier risks.",
        )
    ]

    with pytest.raises(ValueError, match="empty contexts"):
        RagasRunner().run(
            examples,
            report_path=tmp_path / "ragas_report.csv",
            summary_path=tmp_path / "ragas_summary.json",
            debug_dataset_path=tmp_path / "ragas_debug_dataset.json",
        )


def test_runner_fails_when_contexts_are_not_list_of_strings(tmp_path) -> None:
    examples = [
        EvaluationExample(
            question="What supply chain or supplier risks does Apple disclose?",
            ground_truth="Apple discloses supplier and supply chain risks.",
            contexts='["Apple supplier risks"]',
            answer="Apple discloses supplier risks.",
        )
    ]

    with pytest.raises(TypeError, match=r"contexts must be list\[str\]"):
        RagasRunner().run(
            examples,
            report_path=tmp_path / "ragas_report.csv",
            summary_path=tmp_path / "ragas_summary.json",
            debug_dataset_path=tmp_path / "ragas_debug_dataset.json",
        )


def test_runner_fails_when_answer_or_ground_truth_is_empty(tmp_path) -> None:
    invalid_answer = [
        EvaluationExample(
            question="What supply chain or supplier risks does Apple disclose?",
            ground_truth="Apple discloses supplier and supply chain risks.",
            contexts=["Apple relies on suppliers."],
            answer="",
        )
    ]
    invalid_ground_truth = [
        EvaluationExample(
            question="What supply chain or supplier risks does Apple disclose?",
            ground_truth="",
            contexts=["Apple relies on suppliers."],
            answer="Apple discloses supplier risks.",
        )
    ]

    runner = RagasRunner()
    with pytest.raises(ValueError, match="empty answer"):
        runner.run(
            invalid_answer,
            report_path=tmp_path / "answer_report.csv",
            summary_path=tmp_path / "answer_summary.json",
            debug_dataset_path=tmp_path / "answer_debug.json",
        )
    with pytest.raises(ValueError, match="empty ground_truth"):
        runner.run(
            invalid_ground_truth,
            report_path=tmp_path / "truth_report.csv",
            summary_path=tmp_path / "truth_summary.json",
            debug_dataset_path=tmp_path / "truth_debug.json",
        )


def test_apple_ragas_smoke_dataset_scores_non_zero_with_fallback(tmp_path) -> None:
    examples = [
        EvaluationExample(
            question="What supply chain or supplier risks does Apple disclose?",
            ground_truth=(
                "Apple discloses supply constraints, component availability, supplier "
                "performance, logistics disruptions, and reliance on third-party manufacturers."
            ),
            contexts=[
                (
                    "Apple depends on component availability, supplier performance, logistics, "
                    "and third-party manufacturers. Supply constraints can affect product "
                    "availability, costs, and operating results."
                )
            ],
            answer=(
                "Apple discloses risks from component availability, supplier performance, "
                "logistics disruptions, and third-party manufacturers."
            ),
            metadata={"source": "hardcoded_smoke_test", "company": "Apple"},
        )
    ]
    runner = RagasRunner(allow_fallback=True)
    runner._run_ragas = lambda examples: (_ for _ in ()).throw(RuntimeError("no provider"))

    result = runner.run(
        examples,
        run_id="apple-smoke",
        report_path=tmp_path / "ragas_report.csv",
        summary_path=tmp_path / "ragas_summary.json",
        debug_dataset_path=tmp_path / "ragas_debug_dataset.json",
    )

    row = result.rows[0]
    assert row.faithfulness > 0
    assert row.answer_relevancy > 0
    assert row.context_precision > 0
    assert row.context_recall > 0


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

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.router import ReasoningRouter
from app.db.session import SessionLocal
from app.retrieval.providers import get_embedding_provider, get_reranker


DEBUG_DATASET_PATH = Path("evals/reports/ragas_debug_dataset.json")
REPORT_PATH = Path("evals/reports/ragas_report.csv")
FAILURE_ANALYSIS_PATH = Path("evals/reports/ragas_failure_analysis.csv")
BASELINE_AGGREGATE = {
    "faithfulness": 0.5801,
    "answer_relevancy": 0.08,
    "context_precision": 0.4667,
    "context_recall": 0.2349,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose low-scoring RAGAS evaluation rows.")
    parser.add_argument("--debug-dataset-path", type=Path, default=DEBUG_DATASET_PATH)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    parser.add_argument("--output-path", type=Path, default=FAILURE_ANALYSIS_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    debug_rows = json.loads(args.debug_dataset_path.read_text(encoding="utf-8"))
    with args.report_path.open(newline="", encoding="utf-8") as file:
        score_rows = list(csv.DictReader(file))

    joined = []
    for score_row, debug_row in zip(score_rows, debug_rows):
        joined.append(
            {
                "question": score_row["question"],
                "answer_relevancy": float(score_row["answer_relevancy"]),
                "context_recall": float(score_row["context_recall"]),
                "context_precision": float(score_row["context_precision"]),
                "faithfulness": float(score_row["faithfulness"]),
                "answer": debug_row["answer"],
                "ground_truth": debug_row["ground_truth"],
                "contexts": debug_row["contexts"],
                "filters": debug_row.get("metadata", {}).get("filters", {}),
            }
        )

    failures = sorted(joined, key=lambda row: (row["answer_relevancy"], row["context_recall"]))[
        : args.max_rows
    ]
    aggregate = _aggregate(score_rows)
    output_rows: list[dict[str, Any]] = [
        {
            "row_type": "aggregate_before_after",
            "question": "",
            "before_faithfulness": BASELINE_AGGREGATE["faithfulness"],
            "after_faithfulness": aggregate["faithfulness"],
            "before_answer_relevancy": BASELINE_AGGREGATE["answer_relevancy"],
            "after_answer_relevancy": aggregate["answer_relevancy"],
            "before_context_precision": BASELINE_AGGREGATE["context_precision"],
            "after_context_precision": aggregate["context_precision"],
            "before_context_recall": BASELINE_AGGREGATE["context_recall"],
            "after_context_recall": aggregate["context_recall"],
            "applied_filters": "",
            "similarity_scores": "",
            "retrieved_context_summaries": "",
            "answer": "",
            "ground_truth": "",
        }
    ]

    with SessionLocal() as session:
        router = ReasoningRouter(
            session=session,
            embedding_provider=get_embedding_provider(),
            reranker=get_reranker(),
        )
        for failure in failures:
            result = router.ask(
                query=failure["question"],
                top_k=args.top_k,
                filters=failure["filters"],
            )
            summaries = [
                _single_line(chunk.get("text_preview", ""))[:300]
                for chunk in result.retrieved_chunks
            ]
            scores = [
                str(chunk.get("similarity_score", ""))
                for chunk in result.retrieved_chunks
            ]
            print("\nFAILED RAGAS ROW")
            print(f"question: {failure['question']}")
            print(f"answer_relevancy: {failure['answer_relevancy']}")
            print(f"context_recall: {failure['context_recall']}")
            print(f"answer: {_single_line(failure['answer'])}")
            print(f"ground_truth: {_single_line(failure['ground_truth'])}")
            print(f"applied_filters: {result.applied_filters}")
            print("retrieved_context_summaries:")
            for index, summary in enumerate(summaries, start=1):
                print(f"  {index}. score={scores[index - 1]} {summary}")

            output_rows.append(
                {
                    "row_type": "failure",
                    "question": failure["question"],
                    "before_faithfulness": "",
                    "after_faithfulness": failure["faithfulness"],
                    "before_answer_relevancy": "",
                    "after_answer_relevancy": failure["answer_relevancy"],
                    "before_context_precision": "",
                    "after_context_precision": failure["context_precision"],
                    "before_context_recall": "",
                    "after_context_recall": failure["context_recall"],
                    "applied_filters": json.dumps(result.applied_filters),
                    "similarity_scores": json.dumps(scores),
                    "retrieved_context_summaries": json.dumps(summaries),
                    "answer": failure["answer"],
                    "ground_truth": failure["ground_truth"],
                }
            )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"\nFailure analysis CSV: {args.output_path}")


def _single_line(text: str) -> str:
    return " ".join(str(text).split())


def _aggregate(rows: list[dict[str, str]]) -> dict[str, float]:
    metrics = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    return {
        metric: round(sum(float(row[metric]) for row in rows) / len(rows), 4)
        for metric in metrics
    }


if __name__ == "__main__":
    main()

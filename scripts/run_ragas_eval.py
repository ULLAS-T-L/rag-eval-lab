from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from app.agents.router import ReasoningRouter
from app.db.session import SessionLocal
from app.evaluation.datasets import EvaluationDatasetBuilder
from app.evaluation.ragas_runner import REPORT_PATH, SUMMARY_PATH, RagasRunner
from app.retrieval.embeddings import PlaceholderEmbeddingProvider
from app.retrieval.rerankers import NoOpReranker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation for the RAG pipeline.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--synthetic-limit", type=int, default=15)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    parser.add_argument("--summary-path", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--no-fallback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with SessionLocal() as session:
        router = ReasoningRouter(
            session=session,
            embedding_provider=PlaceholderEmbeddingProvider(),
            reranker=NoOpReranker(),
        )

        def answer_fn(question: str, filters: dict[str, Any]) -> tuple[str, list[str]]:
            result = router.ask(query=question, top_k=args.top_k, filters=filters)
            contexts = [
                chunk.get("text_preview", "")
                for chunk in result.retrieved_chunks
                if chunk.get("text_preview")
            ]
            return result.answer, contexts

        examples = EvaluationDatasetBuilder(session=session, answer_fn=answer_fn).build(
            synthetic_limit=args.synthetic_limit
        )
        run = RagasRunner(
            session=session,
            allow_fallback=not args.no_fallback,
        ).run(
            examples,
            report_path=args.report_path,
            summary_path=args.summary_path,
        )

    print(f"RAGAS run_id: {run.run_id}")
    print(f"Questions evaluated: {len(run.rows)}")
    print(f"CSV report: {run.report_path}")
    print(f"Aggregate summary: {run.summary_path}")
    print(f"Average scores: {run.aggregate_report.averages}")


if __name__ == "__main__":
    main()

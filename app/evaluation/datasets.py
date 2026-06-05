from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document


@dataclass(frozen=True)
class EvaluationExample:
    question: str
    ground_truth: str
    contexts: list[str]
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_schema(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "contexts": self.contexts,
            "answer": self.answer,
        }


@dataclass(frozen=True)
class BenchmarkQuestion:
    question: str
    ground_truth: str
    filters: dict[str, Any] = field(default_factory=dict)


MANUAL_BENCHMARK_QUESTIONS: tuple[BenchmarkQuestion, ...] = (
    BenchmarkQuestion(
        question="What supply chain or supplier risks does Apple disclose?",
        ground_truth=(
            "Apple discloses that supply constraints, component availability, supplier "
            "performance, logistics disruptions, and reliance on third-party manufacturers can "
            "affect product availability, costs, and operating results."
        ),
        filters={"company": "Apple", "source_file": "Apple_10k.pdf"},
    ),
    BenchmarkQuestion(
        question="What risks does Kirkland describe around fulfillment and logistics capacity?",
        ground_truth=(
            "Kirkland describes risks from supply chain disruption, fulfillment capacity, "
            "inventory planning, distribution and logistics constraints, third-party carriers, "
            "rising costs, and e-commerce order fulfillment failures."
        ),
        filters={"company": "Kirkland", "source_file": "Kirkland_10k.pdf"},
    ),
    BenchmarkQuestion(
        question="What cloud or datacenter risks does Microsoft disclose?",
        ground_truth=(
            "Microsoft discloses that cloud services depend on datacenter capacity, reliable "
            "infrastructure, energy availability, security, supplier performance, and continued "
            "investment to meet demand and maintain service availability."
        ),
        filters={"company": "Microsoft", "source_file": "Microsoft_10k.pdf"},
    ),
)


AnswerFn = Callable[[str, dict[str, Any]], tuple[str, list[str]]]


def empty_answer_fn(question: str, filters: dict[str, Any]) -> tuple[str, list[str]]:
    return "", []


class EvaluationDatasetBuilder:
    def __init__(self, session: Optional[Session] = None, answer_fn: AnswerFn = empty_answer_fn) -> None:
        self.session = session
        self.answer_fn = answer_fn

    def build(
        self,
        *,
        synthetic_limit: int = 15,
        benchmark_questions: Sequence[BenchmarkQuestion] = MANUAL_BENCHMARK_QUESTIONS,
    ) -> list[EvaluationExample]:
        examples: list[EvaluationExample] = []
        examples.extend(self.from_benchmark_questions(benchmark_questions))
        examples.extend(self.from_synthetic_questions(limit=synthetic_limit))
        return examples

    def from_benchmark_questions(
        self,
        benchmark_questions: Sequence[BenchmarkQuestion] = MANUAL_BENCHMARK_QUESTIONS,
    ) -> list[EvaluationExample]:
        examples: list[EvaluationExample] = []
        for benchmark in benchmark_questions:
            answer, contexts = self.answer_fn(benchmark.question, benchmark.filters)
            examples.append(
                EvaluationExample(
                    question=benchmark.question,
                    ground_truth=benchmark.ground_truth,
                    contexts=contexts,
                    answer=answer,
                    metadata={"source": "manual_benchmark", "filters": benchmark.filters},
                )
            )
        return examples

    def from_synthetic_questions(self, *, limit: int = 15) -> list[EvaluationExample]:
        if self.session is None or limit <= 0:
            return []

        statement = (
            select(Chunk, Document)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.synthetic_questions.is_not(None))
            .limit(limit * 3)
        )
        examples: list[EvaluationExample] = []
        for chunk, document in self.session.execute(statement).all():
            questions = [q for q in (chunk.synthetic_questions or []) if isinstance(q, str) and q.strip()]
            if not questions:
                continue
            filters = {
                "company": document.company,
                "source_file": document.source_file,
                "section_title": chunk.section_path,
            }
            answer, contexts = self.answer_fn(questions[0], {k: v for k, v in filters.items() if v})
            examples.append(
                EvaluationExample(
                    question=questions[0],
                    ground_truth=chunk.summary or chunk.content,
                    contexts=contexts,
                    answer=answer,
                    metadata={
                        "source": "synthetic_question",
                        "chunk_id": str(chunk.id),
                        "document_id": str(document.id),
                        "filters": filters,
                    },
                )
            )
            if len(examples) >= limit:
                break
        return examples


def to_ragas_dataset(examples: Sequence[EvaluationExample]) -> dict[str, list[Any]]:
    return {
        "question": [example.question for example in examples],
        "ground_truth": [example.ground_truth for example in examples],
        "contexts": [example.contexts for example in examples],
        "answer": [example.answer for example in examples],
    }

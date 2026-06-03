from __future__ import annotations

from app.agents.router import ReasoningRouter
from app.db.models import AnswerLog
from app.evaluation.trulens_tracing import TruLensTracer
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.rerankers import NoOpReranker
from app.retrieval.retriever import Citation, RetrievedChunk


class DeterministicEmbeddingProvider:
    model_name = "deterministic"
    dimensions = 3

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class FakeSession:
    def __init__(self) -> None:
        self.logged = []

    def add(self, item) -> None:
        self.logged.append(item)

    def commit(self) -> None:
        return None

    def get_bind(self):
        return None


def test_trulens_tracer_captures_pipeline_fields_and_feedback_scores() -> None:
    chunk = RetrievedChunk(
        chunk_id="chunk-1",
        chunk_text="Apple relies on suppliers for product availability.",
        similarity_score=0.91,
        document_metadata={"company": "Apple"},
        citations=[
            Citation(
                document_id="doc-1",
                source_file="Apple_10k.pdf",
                page_start=1,
                page_end=2,
                section_title="Risk Factors",
                chunk_id="chunk-1",
            )
        ],
        metadata={"company": "Apple"},
    )

    trace = TruLensTracer().record_pipeline(
        question="What supplier risks does Apple disclose?",
        retrieved_chunks=[chunk],
        generated_answer="Apple relies on suppliers for product availability.",
        citations=[{"chunk_id": "chunk-1"}],
        latency_ms=25,
        retrieval_latency_ms=10,
        generation_latency_ms=5,
        run_id="trace-1",
    )

    metadata = trace.to_metadata()
    assert metadata["trulens_run_id"] == "trace-1"
    assert metadata["question"] == "What supplier risks does Apple disclose?"
    assert metadata["similarity_scores"] == {"chunk-1": 0.91}
    assert metadata["feedback_scores"]["context_relevance"] > 0
    assert metadata["feedback_scores"]["groundedness"] == 1.0
    assert metadata["feedback_scores"]["answer_relevance"] > 0


def test_reasoning_router_stores_trulens_run_id_and_trace_metadata(monkeypatch) -> None:
    import app.agents.router as router_module

    class CompanyRetriever:
        retrieval_strategy = "vector"

        def __init__(self, session, embedding_provider: EmbeddingProvider) -> None:
            pass

        def retrieve(self, query, top_k=5, filters=None, strategy="metadata_then_vector"):
            return [
                RetrievedChunk(
                    chunk_id="chunk-1",
                    chunk_text="Apple supplier disruption may affect product availability.",
                    similarity_score=0.9,
                    document_metadata={"company": "Apple"},
                    citations=[
                        Citation(
                            document_id="doc-1",
                            source_file="Apple_10k.pdf",
                            page_start=10,
                            page_end=11,
                            section_title="Risk Factors",
                            chunk_id="chunk-1",
                        )
                    ],
                    metadata={"company": "Apple"},
                )
            ]

    monkeypatch.setattr(router_module, "VectorRetriever", CompanyRetriever)
    session = FakeSession()

    result = ReasoningRouter(
        session=session,
        embedding_provider=DeterministicEmbeddingProvider(),
        reranker=NoOpReranker(),
    ).ask("What supplier risks does Apple disclose?", top_k=1, filters={"company": "Apple"})

    answer_logs = [item for item in session.logged if isinstance(item, AnswerLog)]
    assert result.trulens_run_id
    assert result.observability_scores["groundedness"] > 0
    assert len(answer_logs) == 1
    assert answer_logs[0].trulens_run_id == result.trulens_run_id
    assert answer_logs[0].answer_metadata["trulens"]["trulens_run_id"] == result.trulens_run_id
    assert answer_logs[0].answer_metadata["trulens"]["retrieval_latency_ms"] >= 0
    assert answer_logs[0].answer_metadata["trulens"]["generation_latency_ms"] >= 0

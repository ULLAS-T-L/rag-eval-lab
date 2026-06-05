from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

import app.api.query as query_api
import app.agents.router as router_module
from app.agents.answer_generator import (
    GroundedAnswerGenerator,
    INSUFFICIENT_EVIDENCE_MESSAGE,
)
from app.agents.planner import QueryPlanner
from app.agents.router import ReasoningRouter
from app.db.models import AnswerLog
from app.db.session import get_db
from app.main import app
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.rerankers import NoOpReranker
from app.retrieval.retriever import (
    Citation,
    RetrievedChunk,
    RetrievalFilters,
    RetrievalService,
    VectorRetriever,
)


class DeterministicEmbeddingProvider:
    model_name = "deterministic"
    dimensions = 3

    def embed_query(self, text: str) -> list[float]:
        if "apple" in text.lower():
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class InMemoryVectorRetriever:
    retrieval_strategy = "vector"

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self.embedding_provider = embedding_provider
        self.rows = [
            ("1", "Apple revenue grew.", [1.0, 0.0, 0.0], {"source_file": "apple_10k.pdf"}),
            ("2", "Microsoft cloud revenue grew.", [0.0, 1.0, 0.0], {"source_file": "microsoft_10k.pdf"}),
        ]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: RetrievalFilters = None,
        strategy: str = "metadata_then_vector",
    ) -> list[RetrievedChunk]:
        query_vector = self.embedding_provider.embed_query(query)
        scored = []
        for chunk_id, text, vector, metadata in self.rows:
            if filters and filters.source_file and metadata["source_file"] != filters.source_file:
                continue
            score = sum(left * right for left, right in zip(query_vector, vector))
            scored.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    chunk_text=text,
                    similarity_score=score,
                    document_metadata={"title": metadata["source_file"]},
                    citations=[],
                    metadata=metadata,
                )
            )
        return sorted(scored, key=lambda chunk: chunk.similarity_score, reverse=True)[:top_k]


class FakeSession:
    def __init__(self) -> None:
        self.logged = []

    def add(self, item) -> None:
        self.logged.append(item)

    def commit(self) -> None:
        return None

    def get_bind(self):
        return None


def test_retrieval_ranking() -> None:
    service = RetrievalService(
        session=FakeSession(),
        retriever=InMemoryVectorRetriever(DeterministicEmbeddingProvider()),
        reranker=NoOpReranker(),
    )

    chunks, latency_ms = service.retrieve("apple revenue", top_k=2)

    assert latency_ms >= 0
    assert chunks[0].chunk_id == "1"
    assert chunks[0].similarity_score > chunks[1].similarity_score


def test_metadata_filtering() -> None:
    service = RetrievalService(
        session=FakeSession(),
        retriever=InMemoryVectorRetriever(DeterministicEmbeddingProvider()),
        reranker=NoOpReranker(),
    )

    chunks, _ = service.retrieve(
        "apple revenue",
        top_k=2,
        filters=RetrievalFilters(source_file="microsoft_10k.pdf"),
    )

    assert len(chunks) == 1
    assert chunks[0].metadata["source_file"] == "microsoft_10k.pdf"


def test_pgvector_query_uses_cosine_distance_and_filters() -> None:
    retriever = VectorRetriever(session=object(), embedding_provider=DeterministicEmbeddingProvider())
    statement = retriever.build_query(
        query_vector=[1.0, 0.0, 0.0],
        top_k=3,
        filters=RetrievalFilters(
            document_id=str(uuid4()),
            source_file="apple_10k.pdf",
            company="Apple",
            page=4,
            section_title="Risk Factors",
        ),
    )

    compiled = statement.compile(dialect=postgresql.dialect())
    sql = str(compiled)
    params = compiled.params.values()

    assert "embeddings.vector <=>" in sql
    assert "chunks.document_id" in sql
    assert "documents.company" in sql
    assert "documents.source_file" in sql
    assert "source_file" in params
    assert "page_start" in params
    assert "page_end" in params
    assert "chunks.section_path" in sql
    assert "chunks.section_title" in sql
    assert "LIMIT" in sql


def test_retrieval_filters_can_drop_only_section_title() -> None:
    filters = RetrievalFilters(
        source_file="Apple_10k.pdf",
        company="Apple",
        year=2024,
        page_start=1,
        page_end=10,
        section_title="Risk Factors",
    )

    relaxed = filters.without_section_title()

    assert relaxed.source_file == "Apple_10k.pdf"
    assert relaxed.company == "Apple"
    assert relaxed.year == 2024
    assert relaxed.page_start == 1
    assert relaxed.page_end == 10
    assert relaxed.section_title is None


def test_retrieve_endpoint_response_shape(monkeypatch) -> None:
    class FakeService:
        def __init__(self, session, retriever, reranker) -> None:
            self.retriever = retriever

        def retrieve(self, query, top_k=5, filters=None):
            return [
                RetrievedChunk(
                    chunk_id="chunk-1",
                    chunk_text="Revenue increased.",
                    similarity_score=0.9,
                    document_metadata={"title": "Apple 10-K"},
                    citations=[
                        Citation(
                            document_id="doc-1",
                            source_file="apple_10k.pdf",
                            page_start=1,
                            page_end=2,
                            section_title="Business",
                            chunk_id="chunk-1",
                        )
                    ],
                    metadata={"source_file": "apple_10k.pdf"},
                )
            ], 12

    class FakeRetriever:
        retrieval_strategy = "vector"

        def __init__(self, session, embedding_provider) -> None:
            pass

    app.dependency_overrides[get_db] = lambda: object()
    monkeypatch.setattr(query_api, "RetrievalService", FakeService)
    monkeypatch.setattr(query_api, "VectorRetriever", FakeRetriever)

    client = TestClient(app)
    response = client.post("/query/retrieve", json={"query": "revenue", "top_k": 1})

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval_strategy"] == "vector"
    assert payload["latency_ms"] == 12
    assert payload["chunks"][0]["chunk_text"] == "Revenue increased."


def test_ask_endpoint_works(monkeypatch) -> None:
    class FakeRouter:
        def __init__(self, session, embedding_provider, reranker) -> None:
            pass

        def ask(self, query, top_k, filters=None):
            return type(
                "Result",
                (),
                {
                    "answer": "Apple cited supply chain constraints. [1]",
                    "citations": [
                        {
                            "source_file": "Apple_10k.pdf",
                            "page_start": 10,
                            "page_end": 12,
                            "section_title": "Risk Factors",
                            "chunk_id": "chunk-1",
                        }
                    ],
                    "retrieved_chunks": [{"chunk_id": "chunk-1"}],
                    "applied_filters": {"company": "Apple"},
                    "retrieval_strategy": "metadata_then_vector",
                    "confidence_score": None,
                    "latency_ms": 10,
                    "token_usage": None,
                },
            )()

    app.dependency_overrides[get_db] = lambda: object()
    monkeypatch.setattr(query_api, "ReasoningRouter", FakeRouter)

    client = TestClient(app)
    response = client.post(
        "/query/ask",
        json={
            "query": "What supply chain risks did Apple mention?",
            "top_k": 5,
            "filters": {"company": "Apple"},
        },
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"].startswith("Apple cited")
    assert payload["retrieval_strategy"] == "metadata_then_vector"
    assert payload["citations"][0]["source_file"] == "Apple_10k.pdf"


def test_planner_uses_vector_only_without_metadata() -> None:
    plan = QueryPlanner().plan("What risks were mentioned?", filters={})

    assert plan.retrieval_strategy == "metadata_then_vector"
    assert plan.filters["section_title"] == "Risk Factors"


def test_planner_infers_company_filter() -> None:
    plan = QueryPlanner().plan("What supply chain risks did Apple mention?", filters={})

    assert plan.retrieval_strategy == "metadata_then_vector"
    assert plan.filters["company"] == "Apple"
    assert plan.filters["source_file"] == "Apple_10k.pdf"


def test_vector_only_retrieval_plan_when_no_filters() -> None:
    plan = QueryPlanner().plan("Describe revenue growth patterns", filters={})

    assert plan.retrieval_strategy == "vector_only"
    assert plan.filters == {}


def test_grounded_answer_includes_citations() -> None:
    chunk = RetrievedChunk(
        chunk_id="chunk-1",
        chunk_text="Apple depends on suppliers and logistics partners for product availability.",
        similarity_score=0.8,
        document_metadata={"title": "Apple 10-K"},
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
        metadata={"source_file": "Apple_10k.pdf"},
    )

    result = GroundedAnswerGenerator().generate("Apple supplier risks", [chunk])

    assert "[1]" in result.answer
    assert result.citations[0]["chunk_id"] == "chunk-1"


def test_insufficient_evidence_returns_safe_response() -> None:
    result = GroundedAnswerGenerator().generate("Apple supplier risks", [])

    assert result.answer == INSUFFICIENT_EVIDENCE_MESSAGE
    assert result.citations == []


def test_reasoning_router_logs_answer_row(monkeypatch) -> None:
    class FakeVectorRetriever:
        retrieval_strategy = "vector"

        def __init__(self, session, embedding_provider) -> None:
            pass

        def retrieve(self, query, top_k=5, filters=None, strategy="metadata_then_vector"):
            return [
                RetrievedChunk(
                    chunk_id="chunk-1",
                    chunk_text="Apple supplier disruptions may affect operations.",
                    similarity_score=0.9,
                    document_metadata={"title": "Apple 10-K"},
                    citations=[
                        Citation(
                            document_id="doc-1",
                            source_file="Apple_10k.pdf",
                            page_start=4,
                            page_end=5,
                            section_title="Risk Factors",
                            chunk_id="chunk-1",
                        )
                    ],
                    metadata={"source_file": "Apple_10k.pdf"},
                )
            ]

    session = FakeSession()
    monkeypatch.setattr(router_module, "VectorRetriever", FakeVectorRetriever)

    result = ReasoningRouter(
        session=session,
        embedding_provider=DeterministicEmbeddingProvider(),
        reranker=NoOpReranker(),
    ).ask("What supply chain risks did Apple mention?", top_k=1, filters={"company": "Apple"})

    answer_logs = [item for item in session.logged if isinstance(item, AnswerLog)]
    assert result.answer != INSUFFICIENT_EVIDENCE_MESSAGE
    assert len(answer_logs) == 1
    assert answer_logs[0].retrieved_chunk_ids == ["chunk-1"]
    assert answer_logs[0].applied_filters["company"] == "Apple"


def test_ask_company_filter_returns_only_apple_chunks(monkeypatch) -> None:
    class CompanyFilteringRetriever:
        retrieval_strategy = "vector"

        def __init__(self, session, embedding_provider) -> None:
            self.rows = [
                ("apple-1", "Apple supplier disruption risk.", "Apple", "Apple_10k.pdf"),
                ("amazon-1", "Amazon fulfillment network risk.", "Amazon", "Amazon_10k.pdf"),
            ]

        def retrieve(self, query, top_k=5, filters=None, strategy="metadata_then_vector"):
            return [
                RetrievedChunk(
                    chunk_id=chunk_id,
                    chunk_text=text,
                    similarity_score=0.8,
                    document_metadata={"company": company, "source_file": source_file},
                    citations=[
                        Citation(
                            document_id=f"doc-{company}",
                            source_file=source_file,
                            page_start=1,
                            page_end=2,
                            section_title="Risk Factors",
                            chunk_id=chunk_id,
                        )
                    ],
                    metadata={"company": company, "source_file": source_file},
                )
                for chunk_id, text, company, source_file in self.rows
                if not filters.company or company == filters.company
            ][:top_k]

    monkeypatch.setattr(router_module, "VectorRetriever", CompanyFilteringRetriever)

    result = ReasoningRouter(
        session=FakeSession(),
        embedding_provider=DeterministicEmbeddingProvider(),
        reranker=NoOpReranker(),
    ).ask("What supply chain risks were mentioned?", top_k=5, filters={"company": "Apple"})

    assert result.retrieved_chunks
    assert all(chunk["metadata"]["company"] == "Apple" for chunk in result.retrieved_chunks)


def test_ask_company_filter_microsoft_excludes_apple_and_amazon(monkeypatch) -> None:
    class CompanyFilteringRetriever:
        retrieval_strategy = "vector"

        def __init__(self, session, embedding_provider) -> None:
            self.rows = [
                ("apple-1", "Apple supplier disruption risk.", "Apple", "Apple_10k.pdf"),
                ("amazon-1", "Amazon fulfillment network risk.", "Amazon", "Amazon_10k.pdf"),
                ("microsoft-1", "Microsoft datacenter supply risk.", "Microsoft", "Microsoft_10k.pdf"),
            ]

        def retrieve(self, query, top_k=5, filters=None, strategy="metadata_then_vector"):
            return [
                RetrievedChunk(
                    chunk_id=chunk_id,
                    chunk_text=text,
                    similarity_score=0.8,
                    document_metadata={"company": company, "source_file": source_file},
                    citations=[
                        Citation(
                            document_id=f"doc-{company}",
                            source_file=source_file,
                            page_start=1,
                            page_end=2,
                            section_title="Risk Factors",
                            chunk_id=chunk_id,
                        )
                    ],
                    metadata={"company": company, "source_file": source_file},
                )
                for chunk_id, text, company, source_file in self.rows
                if not filters.company or company == filters.company
            ][:top_k]

    monkeypatch.setattr(router_module, "VectorRetriever", CompanyFilteringRetriever)

    result = ReasoningRouter(
        session=FakeSession(),
        embedding_provider=DeterministicEmbeddingProvider(),
        reranker=NoOpReranker(),
    ).ask("What risks were mentioned?", top_k=5, filters={"company": "Microsoft"})

    returned_companies = {chunk["metadata"]["company"] for chunk in result.retrieved_chunks}
    assert returned_companies == {"Microsoft"}

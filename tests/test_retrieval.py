from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

import app.api.query as query_api
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
            page=4,
            section_title="Risk Factors",
        ),
    )

    compiled = statement.compile(dialect=postgresql.dialect())
    sql = str(compiled)
    params = compiled.params.values()

    assert "embeddings.vector <=>" in sql
    assert "chunks.document_id" in sql
    assert "source_file" in params
    assert "page_start" in params
    assert "page_end" in params
    assert "chunks.section_path" in sql
    assert "LIMIT" in sql


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

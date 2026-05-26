from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import SessionLocal
from app.retrieval.embeddings import PlaceholderEmbeddingProvider
from app.retrieval.rerankers import NoOpReranker
from app.retrieval.retriever import RetrievalService, VectorRetriever


def main() -> None:
    query = "annual revenue risk factors"
    with SessionLocal() as session:
        retriever = VectorRetriever(session=session, embedding_provider=PlaceholderEmbeddingProvider())
        service = RetrievalService(session=session, retriever=retriever, reranker=NoOpReranker())
        chunks, latency_ms = service.retrieve(query=query, top_k=5)

    print(f"strategy=vector latency_ms={latency_ms} chunks={len(chunks)}")
    for chunk in chunks:
        citation = chunk.citations[0] if chunk.citations else None
        source = citation.source_file if citation else None
        pages = f"{citation.page_start}-{citation.page_end}" if citation else "unknown"
        print(f"{chunk.similarity_score:.4f} {source} pages={pages} chunk={chunk.chunk_id}")


if __name__ == "__main__":
    main()

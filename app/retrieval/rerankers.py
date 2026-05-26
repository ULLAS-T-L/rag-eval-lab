from __future__ import annotations

from typing import Protocol

from app.retrieval.retriever import RetrievedChunk


class BaseReranker(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        ...


class NoOpReranker:
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return chunks

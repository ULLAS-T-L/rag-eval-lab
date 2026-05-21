from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed_query(self, text: str) -> list[float]:
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...


class PlaceholderEmbeddingProvider:
    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 1536

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

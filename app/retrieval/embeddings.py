from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    model_name: str
    dimensions: int

    def embed_query(self, text: str) -> list[float]:
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...


class PlaceholderEmbeddingProvider:
    model_name = "placeholder-zero-vector"
    dimensions = 1536

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * self.dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

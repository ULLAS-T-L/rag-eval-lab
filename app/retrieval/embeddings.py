from __future__ import annotations

from hashlib import sha256
from math import sqrt
from typing import Protocol


class EmbeddingProvider(Protocol):
    model_name: str
    dimensions: int

    def embed_query(self, text: str) -> list[float]:
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...


class PlaceholderEmbeddingProvider:
    model_name = "placeholder-hash-vector"
    dimensions = 1536

    def embed_query(self, text: str) -> list[float]:
        digest = sha256(text.encode("utf-8")).digest()
        values = [float(digest[index % len(digest)]) / 255.0 for index in range(self.dimensions)]
        norm = sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

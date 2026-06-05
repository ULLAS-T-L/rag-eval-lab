from __future__ import annotations

from hashlib import sha256
from math import sqrt
from typing import Optional, Protocol


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


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model_name: str = "text-embedding-3-small",
        dimensions: int = 1536,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI embeddings require the `openai` package. Install requirements.txt."
            ) from exc

        self.model_name = model_name
        self.dimensions = dimensions
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        cleaned = [text.replace("\n", " ").strip() for text in texts]
        if any(not text for text in cleaned):
            raise ValueError("OpenAI embeddings cannot embed empty strings.")
        kwargs = {
            "model": self.model_name,
            "input": cleaned,
            "encoding_format": "float",
        }
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        response = self._client.embeddings.create(**kwargs)
        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]

from __future__ import annotations

from app.core.config import Settings, get_settings
from app.retrieval.embeddings import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    PlaceholderEmbeddingProvider,
)
from app.retrieval.rerankers import (
    BaseReranker,
    CrossEncoderReranker,
    LexicalReranker,
    NoOpReranker,
)


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    settings = settings or get_settings()
    provider = settings.embedding_provider.lower()
    if provider == "auto":
        provider = "openai" if settings.openai_api_key else "placeholder"
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY.")
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model_name=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
    if provider == "placeholder":
        return PlaceholderEmbeddingProvider()
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {settings.embedding_provider}")


def get_reranker(settings: Settings | None = None) -> BaseReranker:
    settings = settings or get_settings()
    provider = settings.reranker_provider.lower()
    if provider == "cross_encoder":
        return CrossEncoderReranker(model_name=settings.reranker_model)
    if provider == "lexical":
        return LexicalReranker()
    if provider == "noop":
        return NoOpReranker()
    raise ValueError(f"Unsupported RERANKER_PROVIDER: {settings.reranker_provider}")

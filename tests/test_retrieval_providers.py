from __future__ import annotations

import pytest

from app.core.config import Settings
from app.retrieval.embeddings import PlaceholderEmbeddingProvider
from app.retrieval.providers import get_embedding_provider, get_reranker
from app.retrieval.rerankers import LexicalReranker, NoOpReranker


def _settings(**overrides) -> Settings:
    values = {
        "DATABASE_URL": "postgresql+psycopg://rag:rag@localhost:5432/rag_eval_lab",
        "OPENAI_API_KEY": None,
        "EMBEDDING_PROVIDER": "auto",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "EMBEDDING_DIMENSIONS": 1536,
        "RERANKER_PROVIDER": "lexical",
        "RERANKER_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    }
    values.update(overrides)
    return Settings(**values)


def test_auto_embedding_provider_uses_placeholder_without_api_key() -> None:
    provider = get_embedding_provider(_settings())

    assert isinstance(provider, PlaceholderEmbeddingProvider)


def test_openai_embedding_provider_requires_api_key() -> None:
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_embedding_provider(_settings(EMBEDDING_PROVIDER="openai"))


def test_reranker_provider_selects_local_options() -> None:
    assert isinstance(get_reranker(_settings(RERANKER_PROVIDER="lexical")), LexicalReranker)
    assert isinstance(get_reranker(_settings(RERANKER_PROVIDER="noop")), NoOpReranker)

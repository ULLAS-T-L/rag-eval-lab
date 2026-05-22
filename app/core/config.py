from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import field_validator
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "rag-eval-lab"
    app_version: str = "0.1.0"
    environment: str = "development"
    database_url: str = Field(
        default="postgresql+psycopg://rag:rag@localhost:5432/rag_eval_lab",
        alias="DATABASE_URL",
    )
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=1536, alias="EMBEDDING_DIMENSIONS")
    llm_model: str = Field(default="gpt-4.1-mini", alias="LLM_MODEL")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("database_url")
    @classmethod
    def use_psycopg_driver(cls, value: str) -> str:
        if value.startswith("postgresql://"):
            return value.replace("postgresql://", "postgresql+psycopg://", 1)
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()

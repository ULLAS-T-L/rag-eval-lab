from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Document(TimestampMixin, Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    source_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    source_file: Mapped[Optional[str]] = mapped_column(String(512))
    company: Mapped[Optional[str]] = mapped_column(String(255))
    year: Mapped[Optional[int]] = mapped_column(Integer)
    document_type: Mapped[Optional[str]] = mapped_column(String(128))
    title: Mapped[Optional[str]] = mapped_column(String(512))
    checksum: Mapped[Optional[str]] = mapped_column(String(128), unique=True)
    doc_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document")


class Chunk(TimestampMixin, Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    section_path: Mapped[Optional[str]] = mapped_column(String(1024))
    token_count: Mapped[Optional[int]] = mapped_column(Integer)
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    document: Mapped[Document] = relationship(back_populates="chunks")
    embedding: Mapped["Embedding"] = relationship(back_populates="chunk")


class Embedding(TimestampMixin, Base):
    __tablename__ = "embeddings"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id"), nullable=False, unique=True)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    vector: Mapped[list[float]] = mapped_column(Vector(1536), nullable=False)
    embedding_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    chunk: Mapped[Chunk] = relationship(back_populates="embedding")


class RetrievalLog(TimestampMixin, Base):
    __tablename__ = "retrieval_logs"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    retriever_name: Mapped[str] = mapped_column(String(255), nullable=False)
    retrieved_chunk_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    scores: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    top_k: Mapped[Optional[int]] = mapped_column(Integer)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)


class AnswerLog(TimestampMixin, Base):
    __tablename__ = "answer_logs"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    retrieval_log_id: Mapped[Optional[str]] = mapped_column(ForeignKey("retrieval_logs.id"))
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    retrieved_chunk_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    applied_filters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    retrieval_strategy: Mapped[Optional[str]] = mapped_column(String(255))
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    answer_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class EvaluationScore(TimestampMixin, Base):
    __tablename__ = "evaluation_scores"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    answer_log_id: Mapped[Optional[str]] = mapped_column(ForeignKey("answer_logs.id"))
    evaluator: Mapped[str] = mapped_column(String(255), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(255), nullable=False)
    score: Mapped[Optional[int]] = mapped_column(Integer)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class HumanReviewDecision(TimestampMixin, Base):
    __tablename__ = "human_review_decisions"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    answer_log_id: Mapped[Optional[str]] = mapped_column(ForeignKey("answer_logs.id"))
    reviewer: Mapped[Optional[str]] = mapped_column(String(255))
    decision: Mapped[str] = mapped_column(String(64), nullable=False)
    rationale: Mapped[Optional[str]] = mapped_column(Text)
    review_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Iterable
from uuid import UUID

from sqlalchemy import delete, select, text
from sqlalchemy.orm import Session

from app.db.models import (
    AnswerLog,
    Base,
    Chunk,
    Document,
    Embedding,
    EvaluationScore,
    HumanReviewDecision,
    RetrievalLog,
)
from app.ingestion.chunking import ChunkCandidate
from app.ingestion.pdf import RawDocument
from app.retrieval.embeddings import EmbeddingProvider


class IngestionStore:
    def __init__(self, session: Session, embedding_provider: EmbeddingProvider) -> None:
        self.session = session
        self.embedding_provider = embedding_provider

    def initialize_schema(self) -> None:
        connection = self.session.connection()
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        connection.execute(text("SET search_path TO public, extensions"))
        Base.metadata.create_all(bind=connection)
        self._ensure_document_metadata_columns()
        self._ensure_chunk_columns()
        self.session.commit()

    def upsert_document(self, raw_document: RawDocument) -> Document:
        checksum = self._file_checksum(raw_document.source_path)
        existing = self.session.scalar(select(Document).where(Document.checksum == checksum))
        if existing:
            self._apply_document_metadata(existing, raw_document)
            self.session.flush()
            return existing

        document = Document(
            source_uri=str(raw_document.source_path),
            source_file=raw_document.metadata.get("source_file") or raw_document.source_path.name,
            company=raw_document.metadata.get("company"),
            year=raw_document.metadata.get("year"),
            document_type=raw_document.metadata.get("document_type"),
            title=raw_document.metadata.get("title") or raw_document.source_path.stem,
            checksum=checksum,
            total_pages=len(raw_document.pages),
            processing_status="processed",
            doc_metadata=raw_document.metadata,
        )
        self.session.add(document)
        self.session.flush()
        return document

    def _apply_document_metadata(self, document: Document, raw_document: RawDocument) -> None:
        metadata = {**(document.doc_metadata or {}), **raw_document.metadata}
        document.source_file = metadata.get("source_file") or raw_document.source_path.name
        document.company = metadata.get("company")
        document.year = metadata.get("year")
        document.document_type = metadata.get("document_type")
        document.title = metadata.get("title") or document.title or raw_document.source_path.stem
        document.total_pages = len(raw_document.pages)
        document.processing_status = "processed"
        document.doc_metadata = metadata

    def _ensure_document_metadata_columns(self) -> None:
        bind = self.session.get_bind()
        if bind is not None and bind.dialect.name == "postgresql":
            self.session.execute(
                text(
                    "ALTER TABLE documents "
                    "ADD COLUMN IF NOT EXISTS source_file VARCHAR(512), "
                    "ADD COLUMN IF NOT EXISTS company VARCHAR(255), "
                    "ADD COLUMN IF NOT EXISTS year INTEGER, "
                    "ADD COLUMN IF NOT EXISTS document_type VARCHAR(128), "
                    "ADD COLUMN IF NOT EXISTS total_pages INTEGER, "
                    "ADD COLUMN IF NOT EXISTS processing_status VARCHAR(64)"
                )
            )

    def _ensure_chunk_columns(self) -> None:
        bind = self.session.get_bind()
        if bind is not None and bind.dialect.name == "postgresql":
            self.session.execute(
                text(
                    "ALTER TABLE chunks "
                    "ADD COLUMN IF NOT EXISTS chunk_type VARCHAR(32), "
                    "ADD COLUMN IF NOT EXISTS section_title VARCHAR(1024), "
                    "ADD COLUMN IF NOT EXISTS summary TEXT, "
                    "ADD COLUMN IF NOT EXISTS keywords JSON DEFAULT '[]', "
                    "ADD COLUMN IF NOT EXISTS synthetic_questions JSON DEFAULT '[]', "
                    "ADD COLUMN IF NOT EXISTS page_start INTEGER, "
                    "ADD COLUMN IF NOT EXISTS page_end INTEGER, "
                    "ADD COLUMN IF NOT EXISTS metadata_json JSON DEFAULT '{}'"
                )
            )

    def store_chunks(self, chunks: Iterable[ChunkCandidate]) -> list[Chunk]:
        stored_chunks: list[Chunk] = []
        chunk_list = list(chunks)
        vectors = self.embedding_provider.embed_documents([chunk.content for chunk in chunk_list])

        for candidate, vector in zip(chunk_list, vectors):
            chunk = Chunk(
                document_id=UUID(candidate.document_id),
                content=candidate.content,
                chunk_index=candidate.chunk_index,
                section_path=candidate.section_path or candidate.section_title,
                section_title=candidate.section_title,
                chunk_type=candidate.chunk_type,
                summary=candidate.summary,
                keywords=candidate.keywords,
                synthetic_questions=candidate.synthetic_questions,
                token_count=candidate.token_count,
                chunk_metadata=candidate.metadata,
                metadata_json=candidate.metadata,
                page_start=candidate.page_start,
                page_end=candidate.page_end,
            )
            self.session.add(chunk)
            self.session.flush()

            embedding = Embedding(
                chunk_id=chunk.id,
                model=self.embedding_provider.model_name,
                vector=vector,
                embedding_metadata={
                    "dimensions": self.embedding_provider.dimensions,
                    "source_file": candidate.source_file,
                    "page_start": candidate.page_start,
                    "page_end": candidate.page_end,
                },
            )
            self.session.add(embedding)
            stored_chunks.append(chunk)

        return stored_chunks

    def reset_for_reingest(self) -> None:
        self.session.execute(delete(EvaluationScore))
        self.session.execute(delete(HumanReviewDecision))
        self.session.execute(delete(AnswerLog))
        self.session.execute(delete(RetrievalLog))
        self.session.execute(delete(Embedding))
        self.session.execute(delete(Chunk))
        self.session.execute(delete(Document))
        self.session.commit()

    def _file_checksum(self, path: Path) -> str:
        digest = sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

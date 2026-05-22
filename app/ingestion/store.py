from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Iterable
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.db.models import Base, Chunk, Document, Embedding
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
        self.session.commit()

    def upsert_document(self, raw_document: RawDocument) -> Document:
        checksum = self._file_checksum(raw_document.source_path)
        existing = self.session.scalar(select(Document).where(Document.checksum == checksum))
        if existing:
            return existing

        document = Document(
            source_uri=str(raw_document.source_path),
            title=raw_document.metadata.get("title") or raw_document.source_path.stem,
            checksum=checksum,
            doc_metadata=raw_document.metadata,
        )
        self.session.add(document)
        self.session.flush()
        return document

    def store_chunks(self, chunks: Iterable[ChunkCandidate]) -> list[Chunk]:
        stored_chunks: list[Chunk] = []
        chunk_list = list(chunks)
        vectors = self.embedding_provider.embed_documents([chunk.content for chunk in chunk_list])

        for candidate, vector in zip(chunk_list, vectors):
            chunk = Chunk(
                document_id=UUID(candidate.document_id),
                content=candidate.content,
                chunk_index=candidate.chunk_index,
                section_path=candidate.section_title,
                token_count=candidate.token_count,
                chunk_metadata=candidate.metadata,
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

    def _file_checksum(self, path: Path) -> str:
        digest = sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

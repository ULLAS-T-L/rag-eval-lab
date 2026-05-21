from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from app.ingestion.chunking import StructureAwareChunker
from app.ingestion.pdf import PDFIngestionPipeline
from app.ingestion.store import IngestionStore
from app.retrieval.embeddings import EmbeddingProvider


class IngestionPipeline:
    def __init__(
        self,
        session: Session,
        embedding_provider: EmbeddingProvider,
        pdf_loader: PDFIngestionPipeline | None = None,
        chunker: StructureAwareChunker | None = None,
    ) -> None:
        self.session = session
        self.pdf_loader = pdf_loader or PDFIngestionPipeline()
        self.chunker = chunker or StructureAwareChunker()
        self.store = IngestionStore(session=session, embedding_provider=embedding_provider)

    def ingest_path(self, path: Path) -> int:
        self.store.initialize_schema()
        pdf_paths = self._pdf_paths(path)
        total_chunks = 0

        for pdf_path in pdf_paths:
            raw_document = self.pdf_loader.load(pdf_path)
            document = self.store.upsert_document(raw_document)
            chunks = self.chunker.chunk_pages(
                raw_document.pages,
                document_id=str(document.id),
                source_file=raw_document.source_path.name,
                document_metadata=raw_document.metadata,
            )
            self.store.store_chunks(chunks)
            total_chunks += len(chunks)

        self.session.commit()
        return total_chunks

    def _pdf_paths(self, path: Path) -> list[Path]:
        path = path.resolve()
        if path.is_file():
            return [path] if path.suffix.lower() == ".pdf" else []
        return sorted(path.glob("*.pdf"))

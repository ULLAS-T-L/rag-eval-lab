from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session

from app.ingestion.chunking import StructureAwareChunker
from app.ingestion.pdf import PDFIngestionPipeline
from app.ingestion.store import IngestionStore
from app.retrieval.embeddings import EmbeddingProvider


@dataclass
class IngestionReport:
    pdfs_found: int = 0
    documents_processed: int = 0
    pages_parsed: int = 0
    chunks_created: int = 0
    table_chunks_created: int = 0
    text_chunks_created: int = 0
    embeddings_stored: int = 0
    errors: list[str] = field(default_factory=list)


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

    def ingest_path(self, path: Path, *, reset: bool = False) -> IngestionReport:
        self.store.initialize_schema()
        if reset:
            self.store.reset_for_reingest()
        pdf_paths = self._pdf_paths(path)
        report = IngestionReport(pdfs_found=len(pdf_paths))

        for pdf_path in pdf_paths:
            try:
                raw_document = self.pdf_loader.load(pdf_path)
                document = self.store.upsert_document(raw_document)
                chunks = self.chunker.chunk_pages(
                    raw_document.pages,
                    document_id=str(document.id),
                    source_file=raw_document.source_path.name,
                    document_metadata=raw_document.metadata,
                )
                stored_chunks = self.store.store_chunks(chunks)
                report.documents_processed += 1
                report.pages_parsed += len(raw_document.pages)
                report.chunks_created += len(stored_chunks)
                report.table_chunks_created += sum(1 for chunk in stored_chunks if chunk.chunk_type == "table")
                report.text_chunks_created += sum(1 for chunk in stored_chunks if chunk.chunk_type == "text")
                report.embeddings_stored += len(stored_chunks)
            except Exception as exc:  # pragma: no cover - surfaced in CLI/reporting
                self.session.rollback()
                report.errors.append(f"{pdf_path.name}: {exc}")

        self.session.commit()
        return report

    def _pdf_paths(self, path: Path) -> list[Path]:
        path = path.resolve()
        if path.is_file():
            return [path] if path.suffix.lower() == ".pdf" else []
        return sorted(path.glob("*.pdf"))

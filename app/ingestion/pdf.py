from __future__ import annotations

from pathlib import Path

from app.ingestion.document_parser import DocumentParser, ParsedPage, RawDocument
from app.ingestion.metadata_generator import infer_document_metadata_from_filename


class PDFIngestionPipeline:
    """Compatibility wrapper around the richer document parser."""

    def __init__(self, parser: DocumentParser | None = None) -> None:
        self.parser = parser or DocumentParser()

    def load(self, path: Path) -> RawDocument:
        return self.parser.parse(path)

    def _infer_document_metadata(self, path: Path) -> dict[str, object]:
        return infer_document_metadata_from_filename(path.name)

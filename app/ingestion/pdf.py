from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RawDocument:
    source_path: Path
    text: str
    metadata: dict


class PDFIngestionPipeline:
    """PDF ingestion skeleton.

    Replace the placeholder with a parser such as PyMuPDF, Unstructured, or
    LlamaParse once source document requirements are known.
    """

    def load(self, path: Path) -> RawDocument:
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got {path}")
        return RawDocument(source_path=path, text="", metadata={"parser": "placeholder"})

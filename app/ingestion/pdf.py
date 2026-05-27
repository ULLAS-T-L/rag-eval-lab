from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


class PDFParserError(RuntimeError):
    pass


@dataclass(frozen=True)
class ParsedPage:
    page_number: int
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RawDocument:
    source_path: Path
    pages: list[ParsedPage]
    metadata: dict[str, Any]

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)


class PDFIngestionPipeline:
    """Load PDF text and metadata with PyMuPDF."""

    def load(self, path: Path) -> RawDocument:
        path = path.resolve()
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got {path}")

        try:
            import fitz
        except ImportError as exc:
            raise PDFParserError("PyMuPDF is required for PDF ingestion. Install pymupdf.") from exc

        with fitz.open(path) as document:
            metadata = dict(document.metadata or {})
            inferred_metadata = self._infer_document_metadata(path)
            metadata.update(
                {
                    **inferred_metadata,
                    "source_file": path.name,
                    "source_path": str(path),
                    "page_count": document.page_count,
                    "parser": "pymupdf",
                }
            )
            pages = [
                ParsedPage(
                    page_number=page.number + 1,
                    text=page.get_text("text").strip(),
                    metadata={
                        "source_file": path.name,
                        "page_number": page.number + 1,
                    },
                )
                for page in document
            ]

        return RawDocument(source_path=path, pages=pages, metadata=metadata)

    def _infer_document_metadata(self, path: Path) -> dict[str, Any]:
        filename = path.name.lower()
        company_map = {
            "apple": "Apple",
            "microsoft": "Microsoft",
            "amazon": "Amazon",
        }
        company = next((value for key, value in company_map.items() if key in filename), None)
        year_match = re.search(r"\b(20\d{2})\b", filename)
        document_type = "10-K" if "10k" in filename.replace("-", "").replace("_", "") else None
        return {
            "company": company,
            "year": int(year_match.group(1)) if year_match else None,
            "document_type": document_type,
        }

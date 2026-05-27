from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.ingestion.metadata_generator import infer_document_metadata_from_filename
from app.ingestion.table_extractor import TableExtractor


@dataclass(frozen=True)
class ParsedBlock:
    text: str
    page_number: int
    block_type: str = "text"
    font_size: float | None = None
    font_name: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    line_count: int = 0
    is_table_like: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedPage:
    page_number: int
    text: str
    metadata: dict[str, Any]
    blocks: list[ParsedBlock] = field(default_factory=list)
    is_empty: bool = False
    scanned_likely: bool = False


@dataclass(frozen=True)
class RawDocument:
    source_path: Path
    pages: list[ParsedPage]
    metadata: dict[str, Any]

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)


class DocumentParser:
    """Parse PDF pages into rich blocks and page-level metadata."""

    def __init__(self, table_extractor: TableExtractor | None = None) -> None:
        self.table_extractor = table_extractor or TableExtractor()

    def parse(self, path: Path) -> RawDocument:
        path = path.resolve()
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got {path}")

        try:
            import fitz
        except ImportError as exc:  # pragma: no cover - dependency issue
            raise RuntimeError("PyMuPDF is required for PDF ingestion. Install pymupdf.") from exc

        with fitz.open(path) as document:
            metadata = dict(document.metadata or {})
            metadata.update(
                {
                    **infer_document_metadata_from_filename(path.name),
                    "source_file": path.name,
                    "source_path": str(path),
                    "page_count": document.page_count,
                    "parser": "pymupdf",
                }
            )

            pages: list[ParsedPage] = []
            for page in document:
                parsed_page = self._parse_page(page, path.name)
                pages.append(parsed_page)

        return RawDocument(source_path=path, pages=pages, metadata=metadata)

    def _parse_page(self, page: Any, source_file: str) -> ParsedPage:
        page_dict = page.get_text("dict")
        blocks: list[ParsedBlock] = []

        for block in page_dict.get("blocks", []):
            block_type = block.get("type", 0)
            if block_type != 0:
                continue

            block_text = self._extract_block_text(block)
            if not block_text:
                continue

            font_size, font_name = self._extract_font_info(block)
            is_table_like = self.table_extractor.is_table_like(block_text)
            blocks.append(
                ParsedBlock(
                    text=block_text,
                    page_number=page.number + 1,
                    block_type="table" if is_table_like else "text",
                    font_size=font_size,
                    font_name=font_name,
                    bbox=self._to_bbox(block.get("bbox")),
                    line_count=len(block.get("lines", [])),
                    is_table_like=is_table_like,
                    metadata={
                        "source_file": source_file,
                        "page_number": page.number + 1,
                    },
                )
            )

        page_text = "\n\n".join(block.text for block in blocks).strip()
        image_count = len(page.get_images(full=True))
        scanned_likely = not page_text and image_count > 0

        return ParsedPage(
            page_number=page.number + 1,
            text=page_text,
            metadata={
                "source_file": source_file,
                "page_number": page.number + 1,
                "image_count": image_count,
                "block_count": len(blocks),
            },
            blocks=blocks,
            is_empty=not bool(page_text),
            scanned_likely=scanned_likely,
        )

    def _extract_block_text(self, block: dict[str, Any]) -> str:
        lines: list[str] = []
        for line in block.get("lines", []):
            spans = [span.get("text", "").strip() for span in line.get("spans", [])]
            line_text = " ".join(part for part in spans if part)
            if line_text:
                lines.append(line_text)
        return "\n".join(lines).strip()

    def _extract_font_info(self, block: dict[str, Any]) -> tuple[float | None, str | None]:
        font_sizes: list[float] = []
        font_names: list[str] = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if "size" in span:
                    font_sizes.append(float(span["size"]))
                if span.get("font"):
                    font_names.append(str(span["font"]))
        font_size = max(font_sizes) if font_sizes else None
        font_name = font_names[0] if font_names else None
        return font_size, font_name

    def _to_bbox(self, bbox: Any) -> tuple[float, float, float, float] | None:
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return tuple(float(value) for value in bbox)  # type: ignore[return-value]
        return None

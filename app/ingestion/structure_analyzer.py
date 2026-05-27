from __future__ import annotations

from dataclasses import dataclass
from statistics import median
import re
from typing import Any

from app.ingestion.document_parser import ParsedBlock, ParsedPage


SECTION_KEYWORDS = [
    "business",
    "risk factors",
    "management discussion",
    "financial statements",
    "notes to financial statements",
]


@dataclass(frozen=True)
class StructuredBlock:
    text: str
    page_number: int
    block_type: str
    section_title: str
    section_path: str
    section_level: int
    is_heading: bool
    is_table: bool
    font_size: float | None
    metadata: dict[str, Any]


class StructureAnalyzer:
    """Detect headings and attach hierarchy metadata to document blocks."""

    def analyze(self, pages: list[ParsedPage]) -> list[StructuredBlock]:
        blocks: list[StructuredBlock] = []
        section_stack: list[str] = []
        font_sizes = [block.font_size for page in pages for block in page.blocks if block.font_size]
        font_threshold = median(font_sizes) if font_sizes else None

        for page in pages:
            for block in self._page_blocks(page):
                if not block.text.strip():
                    continue

                heading = self._detect_heading(block, font_threshold=font_threshold)
                if heading:
                    section_stack = self._update_stack(section_stack, heading["title"], heading["level"])
                    continue

                section_title = section_stack[-1] if section_stack else "Unknown"
                section_path = " > ".join(section_stack) if section_stack else section_title
                blocks.append(
                    StructuredBlock(
                        text=block.text,
                        page_number=block.page_number,
                        block_type="table" if block.is_table_like or block.block_type == "table" else "text",
                        section_title=section_title,
                        section_path=section_path,
                        section_level=len(section_stack) if section_stack else 0,
                        is_heading=False,
                        is_table=block.is_table_like or block.block_type == "table",
                        font_size=block.font_size,
                        metadata=dict(block.metadata),
                    )
                )

        return blocks

    def _page_blocks(self, page: ParsedPage) -> list[ParsedBlock]:
        if page.blocks:
            return page.blocks
        return [
            ParsedBlock(
                text=paragraph,
                page_number=page.page_number,
                block_type="text",
                metadata=dict(page.metadata),
            )
            for paragraph in self._paragraphs(page.text)
            if paragraph.strip()
        ]

    def _detect_heading(
        self,
        block: ParsedBlock,
        *,
        font_threshold: float | None,
    ) -> dict[str, Any] | None:
        normalized = " ".join(block.text.split())
        if not normalized or len(normalized) > 140:
            return None

        lower = normalized.lower()
        has_keyword = any(keyword in lower for keyword in SECTION_KEYWORDS)
        numbered = bool(re.match(r"^(\d+(\.\d+)*|[IVX]+|Item\s+\d+)[\).\s-]+", normalized, re.I))
        uppercase = normalized.isupper() and len(normalized.split()) <= 12
        short_title = (
            len(normalized.split()) <= 5
            and len(normalized) <= 60
            and normalized[0].isupper()
            and not re.search(r"[.:;!?]$", normalized)
        )
        font_bump = font_threshold is not None and block.font_size is not None and block.font_size >= font_threshold + 1.5

        if not (has_keyword or numbered or uppercase or font_bump or short_title):
            return None

        level = self._heading_level(normalized, block.font_size, font_threshold)
        title = self._normalize_heading(normalized)
        return {"title": title, "level": level}

    def _heading_level(self, text: str, font_size: float | None, font_threshold: float | None) -> int:
        numbering = re.match(r"^(\d+(\.\d+)*|[IVX]+|Item\s+\d+)", text, re.I)
        if numbering:
            token = numbering.group(1)
            return min(3, token.count(".") + 1)
        if font_size is not None and font_threshold is not None:
            if font_size >= font_threshold + 3:
                return 1
            if font_size >= font_threshold + 1.5:
                return 2
        return 1

    def _normalize_heading(self, text: str) -> str:
        text = re.sub(r"^\s*(Item\s+\d+(\.\d+)?|(\d+(\.\d+)*)|[IVX]+)[\).\s-]+", "", text, flags=re.I)
        text = " ".join(text.split())
        if not text:
            return "Unknown"
        return text.title() if text.isupper() else text

    def _update_stack(self, stack: list[str], title: str, level: int) -> list[str]:
        if level <= 1:
            return [title]
        base = stack[: level - 1]
        return [*base, title]

    def _paragraphs(self, text: str) -> list[str]:
        return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]

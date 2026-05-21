from __future__ import annotations

from dataclasses import dataclass
import re

from app.ingestion.pdf import ParsedPage


@dataclass(frozen=True)
class ChunkCandidate:
    content: str
    document_id: str
    source_file: str
    page_start: int
    page_end: int
    section_title: str | None
    chunk_index: int
    token_count: int
    metadata: dict


@dataclass(frozen=True)
class _TextBlock:
    text: str
    page_number: int
    section_title: str | None


class StructureAwareChunker:
    def __init__(self, max_tokens: int = 800) -> None:
        self.max_tokens = max_tokens

    def chunk_pages(
        self,
        pages: list[ParsedPage],
        *,
        document_id: str,
        source_file: str,
        document_metadata: dict | None = None,
    ) -> list[ChunkCandidate]:
        blocks = self._extract_blocks(pages)
        if not blocks:
            return []

        chunks: list[ChunkCandidate] = []
        current_text: list[str] = []
        current_pages: list[int] = []
        current_section: str | None = None

        for block in blocks:
            block_tokens = self._count_tokens(block.text)
            current_tokens = self._count_tokens("\n\n".join(current_text))

            if current_text and current_tokens + block_tokens > self.max_tokens:
                chunks.append(
                    self._build_chunk(
                        content="\n\n".join(current_text),
                        pages=current_pages,
                        section_title=current_section,
                        document_id=document_id,
                        source_file=source_file,
                        chunk_index=len(chunks),
                        document_metadata=document_metadata or {},
                    )
                )
                current_text = []
                current_pages = []

            current_text.append(block.text)
            current_pages.append(block.page_number)
            current_section = block.section_title or current_section

        if current_text:
            chunks.append(
                self._build_chunk(
                    content="\n\n".join(current_text),
                    pages=current_pages,
                    section_title=current_section,
                    document_id=document_id,
                    source_file=source_file,
                    chunk_index=len(chunks),
                    document_metadata=document_metadata or {},
                )
            )

        return chunks

    def chunk(self, text: str, metadata: dict | None = None) -> list[ChunkCandidate]:
        source_file = (metadata or {}).get("source_file", "unknown")
        page = ParsedPage(page_number=1, text=text, metadata={"source_file": source_file})
        return self.chunk_pages(
            [page],
            document_id=(metadata or {}).get("document_id", "unpersisted"),
            source_file=source_file,
            document_metadata=metadata or {},
        )

    def _extract_blocks(self, pages: list[ParsedPage]) -> list[_TextBlock]:
        blocks: list[_TextBlock] = []
        current_section: str | None = None

        for page in pages:
            for paragraph in self._paragraphs(page.text):
                if self._looks_like_heading(paragraph):
                    current_section = paragraph.strip()
                    continue
                blocks.append(
                    _TextBlock(
                        text=paragraph,
                        page_number=page.page_number,
                        section_title=current_section,
                    )
                )

        return blocks

    def _build_chunk(
        self,
        *,
        content: str,
        pages: list[int],
        section_title: str | None,
        document_id: str,
        source_file: str,
        chunk_index: int,
        document_metadata: dict,
    ) -> ChunkCandidate:
        token_count = self._count_tokens(content)
        metadata = {
            "document_id": document_id,
            "source_file": source_file,
            "page_start": min(pages),
            "page_end": max(pages),
            "section_title": section_title,
            "chunk_index": chunk_index,
            "token_count": token_count,
            "document_metadata": document_metadata,
        }
        return ChunkCandidate(
            content=content,
            document_id=document_id,
            source_file=source_file,
            page_start=min(pages),
            page_end=max(pages),
            section_title=section_title,
            chunk_index=chunk_index,
            token_count=token_count,
            metadata=metadata,
        )

    def _paragraphs(self, text: str) -> list[str]:
        return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]

    def _looks_like_heading(self, text: str) -> bool:
        normalized = " ".join(text.strip().split())
        if not normalized or len(normalized) > 120:
            return False
        if re.match(r"^(\d+(\.\d+)*|[A-Z])[\).\s-]+[A-Z][\w\s,:-]+$", normalized):
            return True
        if normalized.isupper() and len(normalized.split()) <= 12:
            return True
        return bool(re.match(r"^[A-Z][A-Za-z0-9,: -]{2,80}$", normalized))

    def _count_tokens(self, text: str) -> int:
        return len(re.findall(r"\S+", text))

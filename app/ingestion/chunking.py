from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from app.ingestion.chunk_quality import ChunkQualityEvaluator
from app.ingestion.document_parser import ParsedPage
from app.ingestion.metadata_generator import DeterministicMetadataGenerator
from app.ingestion.question_generator import DeterministicQuestionGenerator
from app.ingestion.structure_analyzer import StructureAnalyzer, StructuredBlock
from app.ingestion.table_extractor import TableExtractor


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
    chunk_type: str = "text"
    section_path: str | None = None
    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    synthetic_questions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class StructureAwareChunker:
    def __init__(
        self,
        max_tokens: int = 800,
        overlap_tokens: int = 50,
        structure_analyzer: StructureAnalyzer | None = None,
        table_extractor: TableExtractor | None = None,
        metadata_generator: DeterministicMetadataGenerator | None = None,
        question_generator: DeterministicQuestionGenerator | None = None,
        chunk_quality: ChunkQualityEvaluator | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.structure_analyzer = structure_analyzer or StructureAnalyzer()
        self.table_extractor = table_extractor or TableExtractor()
        self.metadata_generator = metadata_generator or DeterministicMetadataGenerator(
            question_generator=question_generator
        )
        self.chunk_quality = chunk_quality or ChunkQualityEvaluator()

    def chunk_pages(
        self,
        pages: list[ParsedPage],
        *,
        document_id: str,
        source_file: str,
        document_metadata: dict | None = None,
    ) -> list[ChunkCandidate]:
        structured_blocks = self.structure_analyzer.analyze(pages)
        if not structured_blocks:
            return []

        chunks: list[ChunkCandidate] = []
        buffer: list[StructuredBlock] = []
        current_section = structured_blocks[0].section_title
        current_path = structured_blocks[0].section_path

        def flush_buffer() -> None:
            nonlocal buffer, chunks
            if not buffer:
                return
            pages_range = [block.page_number for block in buffer]
            content = "\n\n".join(block.text for block in buffer).strip()
            section_title = buffer[-1].section_title or current_section
            section_path = buffer[-1].section_path or current_path
            if not content:
                buffer = []
                return

            if self._buffer_is_table(buffer):
                chunks.append(
                    self._build_candidate(
                        content=content,
                        document_id=document_id,
                        source_file=source_file,
                        pages=pages_range,
                        section_title=section_title,
                        section_path=section_path,
                        chunk_index=len(chunks),
                        chunk_type="table",
                        document_metadata=document_metadata or {},
                    )
                )
            else:
                for segment in self._split_text(content):
                    chunks.append(
                        self._build_candidate(
                            content=segment,
                            document_id=document_id,
                            source_file=source_file,
                            pages=pages_range,
                            section_title=section_title,
                            section_path=section_path,
                            chunk_index=len(chunks),
                            chunk_type="text",
                            document_metadata=document_metadata or {},
                        )
                    )
            buffer = []

        for block in structured_blocks:
            if block.is_heading:
                flush_buffer()
                current_section = block.section_title
                current_path = block.section_path
                continue

            if block.block_type == "table":
                flush_buffer()
                chunks.append(
                    self._build_candidate(
                        content=block.text,
                        document_id=document_id,
                        source_file=source_file,
                        pages=[block.page_number],
                        section_title=block.section_title,
                        section_path=block.section_path,
                        chunk_index=len(chunks),
                        chunk_type="table",
                        document_metadata=document_metadata or {},
                    )
                )
                continue

            prospective = buffer + [block]
            if buffer and self._count_tokens("\n\n".join(item.text for item in prospective)) > self.max_tokens:
                flush_buffer()
            buffer.append(block)

        flush_buffer()
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

    def _buffer_is_table(self, blocks: list[StructuredBlock]) -> bool:
        return any(block.is_table or self.table_extractor.is_table_like(block.text) for block in blocks)

    def _split_text(self, text: str) -> list[str]:
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
        if not sentences:
            return [text.strip()] if text.strip() else []

        segments: list[str] = []
        current: list[str] = []
        current_tokens = 0
        previous_tail = ""

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            if sentence_tokens > self.max_tokens:
                if current:
                    segment = " ".join(current).strip()
                    if previous_tail:
                        segment = f"{previous_tail} {segment}".strip()
                    segments.append(segment)
                    previous_tail = self._tail_text(segment)
                    current = []
                    current_tokens = 0
                for segment in self._split_long_sentence(sentence):
                    if previous_tail:
                        segment = f"{previous_tail} {segment}".strip()
                    segments.append(segment)
                    previous_tail = self._tail_text(segment)
                continue
            if current and current_tokens + sentence_tokens > self.max_tokens:
                segment = " ".join(current).strip()
                if previous_tail:
                    segment = f"{previous_tail} {segment}".strip()
                segments.append(segment)
                previous_tail = self._tail_text(segment)
                current = []
                current_tokens = 0

            current.append(sentence)
            current_tokens += sentence_tokens

        if current:
            segment = " ".join(current).strip()
            if previous_tail:
                segment = f"{previous_tail} {segment}".strip()
            segments.append(segment)

        return segments

    def _split_long_sentence(self, sentence: str) -> list[str]:
        words = sentence.split()
        if not words:
            return []
        segments: list[str] = []
        for index in range(0, len(words), self.max_tokens):
            segments.append(" ".join(words[index : index + self.max_tokens]).strip())
        return [segment for segment in segments if segment]

    def _tail_text(self, text: str) -> str:
        tokens = text.split()
        if not tokens:
            return ""
        return " ".join(tokens[-self.overlap_tokens :]) if self.overlap_tokens else ""

    def _build_candidate(
        self,
        *,
        content: str,
        document_id: str,
        source_file: str,
        pages: list[int],
        section_title: str | None,
        section_path: str | None,
        chunk_index: int,
        chunk_type: str,
        document_metadata: dict,
    ) -> ChunkCandidate:
        token_count = self._count_tokens(content)
        generated = self.metadata_generator.generate_chunk_metadata(
            content,
            company=document_metadata.get("company"),
            section_title=section_title,
            chunk_type=chunk_type,
        )
        quality = self.chunk_quality.evaluate(
            content,
            chunk_type=chunk_type,
            token_count=token_count,
            section_title=section_title,
        )
        metadata = {
            "document_id": document_id,
            "source_file": source_file,
            "page_start": min(pages),
            "page_end": max(pages),
            "section_title": section_title,
            "section_path": section_path,
            "chunk_index": chunk_index,
            "token_count": token_count,
            "chunk_type": chunk_type,
            "quality": {
                "score": quality.score,
                "issues": quality.issues,
            },
            "document_metadata": document_metadata,
        }
        metadata.update(generated)
        return ChunkCandidate(
            content=content,
            document_id=document_id,
            source_file=source_file,
            page_start=min(pages),
            page_end=max(pages),
            section_title=section_title,
            chunk_index=chunk_index,
            token_count=token_count,
            chunk_type=chunk_type,
            section_path=section_path,
            summary=generated.get("summary", ""),
            keywords=list(generated.get("keywords", [])),
            entities=list(generated.get("entities", [])),
            synthetic_questions=list(generated.get("synthetic_questions", [])),
            metadata=metadata,
        )

    def _count_tokens(self, text: str) -> int:
        return len(re.findall(r"\S+", text))

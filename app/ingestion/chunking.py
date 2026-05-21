from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkCandidate:
    content: str
    section_path: str | None
    metadata: dict


class StructureAwareChunker:
    def chunk(self, text: str, metadata: dict | None = None) -> list[ChunkCandidate]:
        if not text:
            return []
        return [ChunkCandidate(content=text, section_path=None, metadata=metadata or {})]

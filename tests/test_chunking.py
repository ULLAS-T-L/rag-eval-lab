from app.ingestion.chunking import StructureAwareChunker
from app.ingestion.pdf import ParsedPage


def test_heading_aware_chunking_preserves_required_metadata() -> None:
    pages = [
        ParsedPage(
            page_number=1,
            text="Introduction\n\nThis is the first paragraph.\n\nThis is the second paragraph.",
            metadata={"source_file": "sample.pdf"},
        ),
        ParsedPage(
            page_number=2,
            text="Methods\n\nThis section describes the retrieval process.",
            metadata={"source_file": "sample.pdf"},
        ),
    ]

    chunks = StructureAwareChunker(max_tokens=8).chunk_pages(
        pages,
        document_id="doc-123",
        source_file="sample.pdf",
        document_metadata={"title": "Sample"},
    )

    assert len(chunks) >= 2
    first = chunks[0]
    assert first.document_id == "doc-123"
    assert first.source_file == "sample.pdf"
    assert first.page_start == 1
    assert first.page_end == 1
    assert first.section_title == "Introduction"
    assert first.chunk_index == 0
    assert first.token_count > 0
    assert first.metadata["document_id"] == "doc-123"
    assert first.metadata["source_file"] == "sample.pdf"
    assert first.metadata["page_start"] == 1
    assert first.metadata["section_title"] == "Introduction"


def test_empty_pages_create_no_chunks() -> None:
    chunks = StructureAwareChunker().chunk_pages(
        [ParsedPage(page_number=1, text="", metadata={})],
        document_id="doc-123",
        source_file="empty.pdf",
    )
    assert chunks == []

from pathlib import Path

from app.ingestion.chunking import StructureAwareChunker
from app.ingestion.document_parser import ParsedBlock
from app.ingestion.pdf import PDFIngestionPipeline
from app.ingestion.pdf import ParsedPage
from app.ingestion.structure_analyzer import StructureAnalyzer
from app.ingestion.table_extractor import TableExtractor


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


def test_pdf_metadata_infers_company_and_document_type() -> None:
    metadata = PDFIngestionPipeline()._infer_document_metadata(Path("apple_10k.pdf"))

    assert metadata["company"] == "Apple"
    assert metadata["year"] is None
    assert metadata["document_type"] == "10-K"


def test_heading_detection_and_section_assignment() -> None:
    pages = [
        ParsedPage(
            page_number=1,
            text="RISK FACTORS\n\nThe company faces market risk.",
            metadata={"source_file": "apple_10k.pdf"},
            blocks=[
                ParsedBlock(
                    text="RISK FACTORS",
                    page_number=1,
                    font_size=18,
                    metadata={"source_file": "apple_10k.pdf"},
                ),
                ParsedBlock(
                    text="The company faces market risk.",
                    page_number=1,
                    font_size=10,
                    metadata={"source_file": "apple_10k.pdf"},
                ),
            ],
        )
    ]

    chunks = StructureAwareChunker(max_tokens=50).chunk_pages(
        pages,
        document_id="doc-123",
        source_file="apple_10k.pdf",
        document_metadata={"company": "Apple"},
    )

    assert chunks
    assert chunks[0].section_title == "Risk Factors"
    assert chunks[0].metadata["section_title"] == "Risk Factors"


def test_table_detection() -> None:
    detector = TableExtractor()
    assert detector.is_table_like("Revenue | Cost\n100 | 50\n200 | 80")


def test_chunk_metadata_exists_and_questions_created() -> None:
    pages = [
        ParsedPage(
            page_number=1,
            text="Business\n\nApple sells devices and services in global markets.",
            metadata={"source_file": "apple_10k.pdf"},
        )
    ]

    chunks = StructureAwareChunker(max_tokens=50).chunk_pages(
        pages,
        document_id="doc-123",
        source_file="apple_10k.pdf",
        document_metadata={"company": "Apple"},
    )

    assert chunks
    chunk = chunks[0]
    assert chunk.metadata["summary"]
    assert chunk.metadata["keywords"]
    assert chunk.synthetic_questions
    assert chunk.metadata["synthetic_questions"]

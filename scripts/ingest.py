from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import SessionLocal
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.providers import get_embedding_provider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDFs into PostgreSQL pgvector.")
    parser.add_argument("--path", type=Path, default=Path("data/raw"), help="PDF file or directory path.")
    parser.add_argument("--reset", action="store_true", help="Clear existing ingestion data before re-ingesting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with SessionLocal() as session:
        pipeline = IngestionPipeline(
            session=session,
            embedding_provider=get_embedding_provider(),
        )
        report = pipeline.ingest_path(args.path, reset=args.reset)

    print(f"PDFs found: {report.pdfs_found}")
    print(f"Documents processed: {report.documents_processed}")
    print(f"Pages parsed: {report.pages_parsed}")
    print(f"Chunks created: {report.chunks_created}")
    print(f"Table chunks created: {report.table_chunks_created}")
    print(f"Text chunks created: {report.text_chunks_created}")
    print(f"Embeddings stored: {report.embeddings_stored}")
    print(f"Errors: {len(report.errors)}")
    for error in report.errors:
        print(f"  - {error}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import SessionLocal
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.embeddings import PlaceholderEmbeddingProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDFs into PostgreSQL pgvector.")
    parser.add_argument("--path", type=Path, default=Path("data/raw"), help="PDF file or directory path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with SessionLocal() as session:
        pipeline = IngestionPipeline(
            session=session,
            embedding_provider=PlaceholderEmbeddingProvider(),
        )
        chunk_count = pipeline.ingest_path(args.path)
    print(f"Ingested {chunk_count} chunks from {args.path}")


if __name__ == "__main__":
    main()

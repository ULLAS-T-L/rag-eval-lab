# rag-eval-lab

Production-grade Agentic RAG skeleton for retrieval, evaluation, human review, and red-team testing.

## Stack

- Python 3.12
- FastAPI
- PostgreSQL with pgvector
- SQLAlchemy
- LangGraph
- RAGAS
- TruLens
- Docker Compose

## Project Layout

```text
app/
  api/             FastAPI routes
  ingestion/       PDF loading and chunking pipeline interfaces
  retrieval/       Embedding and retriever interfaces
  agents/          LangGraph workflow placeholders
  evaluation/      RAGAS and TruLens integration placeholders
  human_review/    Human review routing
  redteam/         Red-team scenario definitions
  db/              SQLAlchemy session, models, and schema notes
  core/            Settings and shared config
data/
  raw/
  processed/
evals/
tests/
docker/
scripts/
```

## Setup

1. Create a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Configure environment variables.

```powershell
Copy-Item .env.example .env
```

4. Run the API locally.

```powershell
uvicorn app.main:app --reload
```

5. Check health.

```powershell
Invoke-RestMethod http://localhost:8000/health
```

## Docker

```powershell
docker compose up --build
```

The API is exposed at `http://localhost:8000`, and PostgreSQL with pgvector is exposed at `localhost:5432`.

## Tests

```powershell
python -m pytest
```

Current tests cover the FastAPI health endpoint, database configuration, chunking, and chunk metadata creation.

## Ingestion

Put source PDFs in `data/raw`, start PostgreSQL, then run:

```powershell
python scripts/ingest.py --path data/raw
```

The ingestion pipeline:

- Loads PDFs with PyMuPDF.
- Extracts document metadata and page-level text.
- Creates heading-aware chunks with page spans and token counts.
- Preserves chunk metadata including `document_id`, `source_file`, `page_start`, `page_end`, `section_title`, `chunk_index`, and `token_count`.
- Generates embeddings through the pluggable `EmbeddingProvider` interface.
- Stores documents, chunks, metadata, and embeddings in PostgreSQL with pgvector.

The default embedding provider is a placeholder zero-vector provider so the skeleton remains runnable before a real embedding service is configured.

## Next Implementation Steps

- Add Alembic migrations from `app/db/models.py`.
- Add a real embedding provider implementation.
- Expand structure-aware chunking for tables and page references.
- Implement pgvector similarity retrieval.
- Convert `app/agents/workflow.py` placeholders into a compiled LangGraph graph.
- Add RAGAS datasets and TruLens instrumentation.
- Expand red-team scenarios into automated regression tests.

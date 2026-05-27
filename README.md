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

The default embedding provider is a deterministic placeholder provider so the skeleton remains runnable before a real embedding service is configured.

## Retrieval

Run a direct retrieval benchmark against the configured database:

```powershell
python scripts/test_retrieval.py
```

The API also exposes vector retrieval:

```powershell
Invoke-RestMethod http://localhost:8000/query/retrieve `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"query":"annual revenue risk factors","top_k":5}'
```

Retrieval supports pgvector cosine similarity, metadata filters for `document_id`, `source_file`, `page`, and `section_title`, a no-op reranker interface, and retrieval logging in `retrieval_logs`.

Company and source-file filters are applied before vector ranking. For example, `company = "Apple"` first restricts candidate rows to Apple documents through `documents.company`; if that column is not populated yet, retrieval falls back to matching Apple in `documents.source_file` and chunk source-file metadata. pgvector cosine similarity is then computed only over the filtered candidate chunks.

## Online Query Flow

The online query pipeline is exposed at:

```powershell
Invoke-RestMethod http://localhost:8000/query/ask `
  -Method POST `
  -ContentType "application/json" `
  -Body '{
    "query":"What supply chain risks did Apple mention?",
    "top_k":5,
    "filters":{
      "source_file":null,
      "company":"Apple",
      "year":null,
      "section_title":null,
      "page_start":null,
      "page_end":null
    }
  }'
```

Flow:

1. The planner analyzes the query and merges explicit filters with inferred filters such as company, source file, year, or section.
2. The router chooses `vector_only`, `metadata_then_vector`, or `insufficient_query`.
3. For `metadata_then_vector`, SQL predicates restrict eligible documents and chunks first, then pgvector cosine similarity ranks only those rows.
4. The grounded answer generator answers only from retrieved chunk text and returns citations from retrieved metadata.
5. The answer is logged to `answer_logs` with retrieved chunk IDs, applied filters, retrieval strategy, citations, and latency.

Retrieval, generation, and evaluation are separate responsibilities:

- Retrieval finds and ranks evidence chunks using metadata filters and vector search.
- Generation turns retrieved evidence into a grounded answer with citations and no outside knowledge.
- Evaluation is a later offline/online quality layer for RAGAS, TruLens, and review workflows; it is not implemented in this step.

## Next Implementation Steps

- Add Alembic migrations from `app/db/models.py`.
- Add a real embedding provider implementation.
- Expand structure-aware chunking for tables and page references.
- Convert `app/agents/workflow.py` placeholders into a compiled LangGraph graph.
- Add RAGAS datasets and TruLens instrumentation.
- Expand red-team scenarios into automated regression tests.

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

To clear existing documents, chunks, retrieval logs, and answer logs before a fresh run:

```powershell
python scripts/ingest.py --path data/raw --reset
```

The ingestion pipeline:

```text
PDF
  -> Document Parser
  -> Structure Analyzer
  -> Table Preserver
  -> Boundary Detector
  -> Metadata Generator
  -> Question Generator
  -> Embeddings
  -> PostgreSQL + pgvector
```

The processing stages are:

1. PDF parsing with PyMuPDF page by page.
2. Structure analysis to detect headings and section hierarchy.
3. Table preservation so table-like blocks stay together.
4. Boundary detection and token-aware chunking with overlap.
5. Metadata generation for document-level and chunk-level fields.
6. Synthetic question generation for later evaluation workflows.
7. Embedding creation through the pluggable `EmbeddingProvider` interface.
8. Storage in PostgreSQL with pgvector.

Document records now store `company`, `year`, `document_type`, `source_file`, `total_pages`, and `processing_status`.
Chunk records now store `chunk_type`, `section_title`, `summary`, `keywords`, `synthetic_questions`, `token_count`, `page_start`, `page_end`, and chunk metadata JSON.

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
5. TruLens-style tracing captures the question, retrieved chunks, similarity scores, generated answer, citations, retrieval latency, generation latency, and end-to-end latency.
6. The answer is logged to `answer_logs` with a `trulens_run_id`, retrieved chunk IDs, applied filters, retrieval strategy, citations, latency, and observability scores.

Retrieval, generation, and evaluation are separate responsibilities:

- Retrieval finds and ranks evidence chunks using metadata filters and vector search.
- Generation turns retrieved evidence into a grounded answer with citations and no outside knowledge.
- Evaluation and observability measure quality with RAGAS and TruLens without changing retrieval or generation behavior.

## Evaluation and Observability

RAGAS and TruLens answer different questions:

- RAGAS is an offline evaluation framework. It scores prepared datasets with `question`, `ground_truth`, `contexts`, and `answer` fields, then reports metrics such as faithfulness, answer relevancy, context precision, and context recall. Use it for benchmark runs, regression testing, and objective comparison across retrievers, prompts, models, or chunking strategies.
- TruLens is an observability and tracing framework. It records live application runs and feedback scores for each answer pipeline execution. Use it to inspect what happened for a specific user question: which chunks were retrieved, what scores they had, what answer was generated, which citations were returned, and how long each step took.

Run RAGAS evaluation:

```powershell
python scripts/run_ragas_eval.py
```

The RAGAS report is exported to `evals/reports/ragas_report.csv`.

Every `/query/ask` call now creates a TruLens trace run id and stores it in `answer_logs.trulens_run_id`. Trace metadata is also stored in `answer_logs.answer_metadata["trulens"]`, including:

- question
- retrieved chunks
- similarity scores
- generated answer
- citations
- retrieval latency
- generation latency
- end-to-end latency
- context relevance
- groundedness
- answer relevance

Launch the TruLens dashboard:

```powershell
python scripts/run_trulens_dashboard.py --port 8501
```

Then open `http://localhost:8501`. If the dashboard fails to start, verify that the local TruLens, pandas, numpy, and streamlit versions are compatible.

## Next Implementation Steps

- Add Alembic migrations from `app/db/models.py`.
- Add a real embedding provider implementation.
- Expand structure-aware chunking for tables and page references.
- Convert `app/agents/workflow.py` placeholders into a compiled LangGraph graph.
- Expand red-team scenarios into automated regression tests.

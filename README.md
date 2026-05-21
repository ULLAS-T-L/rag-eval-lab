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
pytest
```

Current tests cover the FastAPI health endpoint and database configuration behavior.

## Next Implementation Steps

- Add Alembic migrations from `app/db/models.py`.
- Replace PDF ingestion placeholders with a production parser.
- Implement structure-aware chunking for headings, tables, and page references.
- Wire the embedding provider to a real model.
- Implement pgvector similarity retrieval.
- Convert `app/agents/workflow.py` placeholders into a compiled LangGraph graph.
- Add RAGAS datasets and TruLens instrumentation.
- Expand red-team scenarios into automated regression tests.

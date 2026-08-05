# Note RAG

A small, phase-by-phase RAG implementation that follows RAGFlow's core
responsibilities while deliberately simplifying its scale.

## Implemented

- FastAPI application and health endpoint
- Deterministic token counting and token-aware chunking
- PostgreSQL documents, chunks, and ingestion jobs
- Alembic database migrations
- TXT, Markdown, and PDF parsing
- Content-addressed local file storage
- SHA-256 duplicate detection
- Synchronous ingestion with persisted progress and failures
- Batched Gemini embeddings stored in pgvector
- Document indexing status and re-indexing
- Vector and PostgreSQL full-text retrieval
- Document and chunk metadata filters
- Weighted reciprocal-rank hybrid fusion
- Document, chunk, and ingestion-job inspection endpoints

Reranking and generation are intentionally absent until their corresponding
phases.

## Setup

```powershell
Copy-Item .env.example .env
docker compose up -d postgres
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m alembic upgrade head
```

## Run

```powershell
.\.venv\Scripts\note-rag.exe
```

Alternatively:

```powershell
.\.venv\Scripts\python.exe -m uvicorn note_rag.api.app:app --reload
```

Open `http://127.0.0.1:8000/docs` to test the API.

## Upload a document

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/v1/documents `
  -Form @{ file = Get-Item ".\documents\example.pdf" }
```

Supported extensions are `.txt`, `.md`, `.markdown`, and `.pdf`.

## Search indexed chunks

```powershell
$body = @{
  query = "How does retrieval work?"
  mode = "hybrid"
  top_k = 10
  vector_weight = 0.7
  filters = @{
    filenames = @("example.pdf")
    source_metadata = @{ page = 2 }
  }
} | ConvertTo-Json -Depth 4

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/v1/retrieval/search `
  -ContentType "application/json" `
  -Body $body
```

Search modes are `vector`, `keyword`, and `hybrid`. Filters support document
IDs, filenames, media types, and exact source-metadata fields.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check src tests migrations
.\.venv\Scripts\pyright.exe
```

Set `TEST_DATABASE_URL` to run the PostgreSQL integration tests:

```powershell
$env:TEST_DATABASE_URL = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
.\.venv\Scripts\python.exe -m pytest
```

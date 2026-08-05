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
- Injectable reranker with a deterministic lexical baseline
- Duplicate-free, token-budgeted context packages
- Source-labelled context chunks with preserved metadata
- Persisted conversations and ordered message history
- Grounded Gemini chat with source citations
- Regular and server-sent-event streaming chat endpoints
- Document, chunk, and ingestion-job inspection endpoints

Background workers remain intentionally deferred to their corresponding phase.

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

## Build a context package

```powershell
$body = @{
  query = "How does retrieval work?"
  mode = "hybrid"
  candidate_k = 20
  max_chunks = 8
  max_context_tokens = 1200
  rerank = $true
  rerank_weight = 0.7
  filters = @{
    filenames = @("example.pdf")
  }
} | ConvertTo-Json -Depth 4

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/v1/retrieval/context `
  -ContentType "application/json" `
  -Body $body
```

The response contains the rendered context, sequential citation IDs, original
retrieval and reranker scores, exact token usage, truncation state, and source
metadata for every included chunk.

## Chat

```powershell
$body = @{
  query = "How does retrieval work?"
  filters = @{
    filenames = @("example.pdf")
  }
} | ConvertTo-Json -Depth 4

$answer = Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/v1/chat `
  -ContentType "application/json" `
  -Body $body

$answer
```

Send the returned `conversation_id` with a later request to include persisted
history. Use `POST /api/v1/chat/stream` for server-sent events. The stream emits
`metadata`, `delta`, and `done` events; the final event contains citations.

Conversation inspection endpoints:

```text
GET /api/v1/conversations
GET /api/v1/conversations/{conversation_id}
```

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

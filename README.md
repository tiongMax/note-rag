# Note RAG

[![CI](https://github.com/tiongMax/note-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/tiongMax/note-rag/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev/)

A production-oriented retrieval-augmented generation (RAG) application for
uploading documents, searching their contents, and holding grounded
conversations with source citations.

Note RAG combines a FastAPI service, React operator interface, PostgreSQL
full-text search, pgvector similarity search, Gemini embeddings and chat, and a
durable background ingestion worker. It follows the core responsibilities of
larger RAG platforms while keeping the implementation compact enough to study
and extend.

## Architecture

```mermaid
flowchart LR
    User["User"] --> UI["React operator interface"]
    UI --> API["FastAPI application"]

    subgraph Ingestion
        API --> Upload["Document upload"]
        Upload --> Jobs[("PostgreSQL job queue")]
        Jobs --> Worker["Background worker"]
        Worker --> Parse["Parse and token-aware chunk"]
        Parse --> Embed["Gemini embeddings"]
        Embed --> Store[("PostgreSQL + pgvector")]
    end

    subgraph Retrieval_and_generation["Retrieval and generation"]
        API --> Query["Search or chat query"]
        Query --> Hybrid["Hybrid retrieval<br/>vector + full text"]
        Store --> Hybrid
        Hybrid --> Rerank["Rerank and build<br/>token-budgeted context"]
        Rerank --> LLM["Grounded Gemini generation"]
        LLM --> Answer["Answer + source citations"]
        Answer --> API
    end

    API <--> History[("Conversation history")]
```

## Highlights

- **Complete ingestion pipeline** — TXT, Markdown, and PDF parsing,
  content-addressed file storage, SHA-256 duplicate detection, token-aware
  chunking, batched embeddings, and indexing.
- **Durable background processing** — persisted job progress, retry scheduling,
  worker leases, failure inspection, and stale-job recovery.
- **Hybrid retrieval** — pgvector similarity search, PostgreSQL full-text
  search, weighted reciprocal-rank fusion, metadata filters, and an injectable
  reranker.
- **Grounded chat** — duplicate-free context assembly, explicit token budgets,
  persisted conversation history, source citations, and server-sent-event
  streaming.
- **Operator interface** — document upload and inspection, ingestion status,
  re-indexing, deletion, conversation history, source filtering, and citation
  inspection.
- **Production foundations** — bearer authentication, request IDs, structured
  logs, rate limiting, security headers, health probes, Prometheus-compatible
  metrics, Docker, migrations, and CI.

## Technology

| Layer | Technology |
| --- | --- |
| API | Python 3.12, FastAPI, Pydantic |
| Data | PostgreSQL 16, pgvector, SQLAlchemy, Alembic |
| AI | Google Gemini embeddings and chat |
| Frontend | React 19, TypeScript, Vite |
| Documents | pypdf, plain text, Markdown |
| Operations | Docker Compose, GitHub Actions, Prometheus-compatible metrics |
| Quality | pytest, Ruff, Pyright |

## Prerequisites

Choose either Docker Compose for the complete stack or a local development
environment.

### Docker

- Docker Engine with Docker Compose
- A [Google AI Studio API key](https://aistudio.google.com/app/apikey)

### Local development

- Python 3.12+
- Node.js 22+ and npm
- PostgreSQL 16 with the pgvector extension
- A Google Gemini API key

## Quick start with Docker

1. Create the environment file:

   ```bash
   cp .env.example .env
   ```

   On PowerShell:

   ```powershell
   Copy-Item .env.example .env
   ```

2. Set these values in `.env`:

   ```env
   GEMINI_API_KEY=replace-with-your-key
   API_AUTH_TOKEN=replace-with-at-least-24-random-characters
   POSTGRES_PASSWORD=replace-with-a-strong-password
   ```

3. Build and start the stack:

   ```bash
   docker compose up -d --build
   ```

4. Open [http://127.0.0.1:8001](http://127.0.0.1:8001). Enter the configured
   `API_AUTH_TOKEN` when prompted.

Check service health with:

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/health/ready
```

## Local development

Start PostgreSQL:

```bash
docker compose up -d postgres
```

Create a virtual environment and install the project:

```bash
python -m venv .venv
```

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m alembic upgrade head
```

Set `GEMINI_API_KEY` in `.env`, then start the API:

```powershell
.\.venv\Scripts\note-rag.exe
```

The API is available at `http://127.0.0.1:8001`. Interactive API documentation
is available at [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs) outside
production.

For frontend development:

```bash
cd frontend
npm install
npm run dev
```

Vite serves the interface at `http://127.0.0.1:5174` and proxies API requests
to port `8001`. Build the production bundle with `npm run build`.

## Configuration

Configuration is read from environment variables and `.env`. Important options
are listed below; see [`.env.example`](.env.example) for the complete reference.

| Variable | Default | Description |
| --- | --- | --- |
| `APP_ENVIRONMENT` | `development` | Runtime mode; production enables stricter validation and disables API docs. |
| `DATABASE_URL` | PostgreSQL on port `6024` | SQLAlchemy database connection URL. |
| `GEMINI_API_KEY` | — | Gemini API key used for embeddings and chat. |
| `EMBEDDING_MODEL` | `gemini-embedding-2` | Embedding model identifier. |
| `CACHE_ENABLED` | `true` | Master switch for persistent query caches. |
| `EMBEDDING_CACHE_ENABLED` | `true` | Cache normalized exact-query embeddings. |
| `RETRIEVAL_CACHE_ENABLED` | `true` | Cache exact retrieval results by corpus version and search configuration. |
| `EMBEDDING_CACHE_TTL_SECONDS` | `86400` | Query-embedding cache lifetime. |
| `RETRIEVAL_CACHE_TTL_SECONDS` | `3600` | Retrieval-result cache lifetime. |
| `CHAT_MODEL` | `gemini-3.5-flash` | Chat model identifier. |
| `API_AUTH_TOKEN` | empty | Bearer token; required in production and must contain at least 24 characters. |
| `CHUNKING_STRATEGY` | `fixed` | Chunking mode: `fixed` or `recursive`. |
| `CHUNK_SIZE` | `200` | Maximum tokens per chunk. |
| `CHUNK_OVERLAP` | `20` | Token overlap between adjacent chunks. |
| `MAX_UPLOAD_BYTES` | `10485760` | Maximum document size in bytes. |
| `BACKGROUND_WORKER_ENABLED` | `true` | Process ingestion asynchronously when enabled. |
| `CONTEXT_MAX_TOKENS` | `1200` | Default context budget for generation. |
| `ALLOWED_HOSTS` | local hosts | Comma-separated trusted HTTP hosts. |
| `ALLOWED_ORIGINS` | local Vite origins | Comma-separated CORS origins. |

Never commit `.env` or real credentials.

## Usage

The operator interface supports the full workflow. The examples below show the
equivalent API calls. In production, add
`Authorization: Bearer <API_AUTH_TOKEN>` to each `/api/v1` request.

### Upload a document

```bash
curl -X POST http://127.0.0.1:8001/api/v1/documents \
  -F "file=@documents/example.pdf"
```

Supported extensions are `.txt`, `.md`, `.markdown`, and `.pdf`. A new upload
returns `202 Accepted` with a job ID when background processing is enabled.
Duplicate content returns the existing document.

Track an ingestion job:

```text
GET /api/v1/ingestion-jobs/{job_id}
```

The persisted lifecycle is:

```text
QUEUED → PARSING → CHUNKING → EMBEDDING → INDEXING → COMPLETED
```

Failed attempts use exponential backoff. Exhausted jobs remain inspectable as
`FAILED`; stale worker leases return to `QUEUED` after application restart.

### Search indexed content

```bash
curl -X POST http://127.0.0.1:8001/api/v1/retrieval/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does retrieval work?",
    "mode": "hybrid",
    "top_k": 10,
    "vector_weight": 0.7,
    "filters": {
      "filenames": ["example.pdf"]
    }
  }'
```

Search modes are `vector`, `keyword`, and `hybrid`. Filters accept document
IDs, filenames, media types, and exact source-metadata fields.

Responses include `X-Embedding-Cache`, `X-Retrieval-Cache`, and
`X-Corpus-Version` headers. Re-indexing or deleting a document advances the
corpus version and invalidates retrieval results; query embeddings remain
reusable because they do not depend on corpus contents.

### Benchmark the query caches

Apply migrations, index a representative corpus, and replace
`benchmarks/queries.example.txt` with enough distinct, corpus-grounded
questions for the requested workload. Keep the corpus, query trace, machine,
and concurrency fixed across configurations. Increase `RATE_LIMIT_REQUESTS`
above the total benchmark request count.

Run the baseline with both caches disabled:

```powershell
$env:CACHE_ENABLED = "false"
.\.venv\Scripts\python.exe -m note_rag

.\.venv\Scripts\python.exe scripts\benchmark_cache.py `
  --queries benchmarks\queries.example.txt `
  --requests 20 --repeat-rate 0.4 --concurrency 10 `
  --label disabled
```

Restart with only embedding caching:

```powershell
$env:CACHE_ENABLED = "true"
$env:EMBEDDING_CACHE_ENABLED = "true"
$env:RETRIEVAL_CACHE_ENABLED = "false"
.\.venv\Scripts\python.exe -m note_rag

.\.venv\Scripts\python.exe scripts\benchmark_cache.py `
  --queries benchmarks\queries.example.txt `
  --requests 20 --repeat-rate 0.4 --concurrency 10 `
  --label embedding-only --clear-cache `
  --baseline-provider-calls 20
```

Restart with both caches enabled and run the same trace:

```powershell
$env:RETRIEVAL_CACHE_ENABLED = "true"
.\.venv\Scripts\python.exe -m note_rag

.\.venv\Scripts\python.exe scripts\benchmark_cache.py `
  --queries benchmarks\queries.example.txt `
  --requests 20 --repeat-rate 0.4 --concurrency 10 `
  --label embedding-and-retrieval --clear-cache `
  --baseline-provider-calls 20
```

Repeat at `--repeat-rate 0`, `0.2`, `0.4`, and `0.8`, and at the desired
concurrency levels. The fixed default seed makes every matching invocation use
the same trace. Each run writes per-request CSV and aggregate JSON under
`benchmarks/results`, including p50/p95/p99 latency, cold/embedding-warm/fully
warm splits, cache-hit rates, provider calls per query, throughput, and API-call
reduction. Run each configuration three times and report the median result.

For the corpus-update profile, first warm both caches, then upload, re-index, or
delete a document. Run the fixed trace once with caches disabled to capture
post-update ground truth, then run it again with caches enabled and pass the
baseline CSV through `--ground-truth-csv`. The summary reports the fraction of
successful requests whose ordered chunk IDs differ as `stale_result_rate`; it
should be zero. Do not use `--clear-cache` for this profile, because the test is
intended to exercise corpus-version invalidation.

### Chat with citations

```bash
curl -X POST http://127.0.0.1:8001/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize the retrieval approach.",
    "mode": "hybrid",
    "filters": {
      "filenames": ["example.pdf"]
    }
  }'
```

Reuse the returned `conversation_id` in subsequent requests to include
persisted history. Use `POST /api/v1/chat/stream` for server-sent events; the
stream emits `metadata`, `delta`, and `done` events.

## API overview

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Process liveness |
| `GET` | `/health/ready` | Database readiness |
| `GET` | `/metrics` | Prometheus-compatible metrics |
| `POST` | `/api/v1/documents` | Upload and enqueue a document |
| `GET` | `/api/v1/documents` | List documents |
| `GET` | `/api/v1/documents/{id}` | Inspect a document |
| `GET` | `/api/v1/documents/{id}/chunks` | List stored chunks |
| `POST` | `/api/v1/documents/{id}/index` | Re-index a document |
| `DELETE` | `/api/v1/documents/{id}` | Delete a document |
| `GET` | `/api/v1/ingestion-jobs/{id}` | Inspect ingestion progress |
| `POST` | `/api/v1/retrieval/search` | Run keyword, vector, or hybrid search |
| `POST` | `/api/v1/retrieval/context` | Build a reranked context package |
| `POST` | `/api/v1/chat` | Generate a grounded answer |
| `POST` | `/api/v1/chat/stream` | Stream a grounded answer |
| `GET` | `/api/v1/conversations` | List conversations |
| `GET` | `/api/v1/conversations/{id}` | Read conversation history |

## Testing and quality

Run the backend checks:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check src tests migrations
.\.venv\Scripts\pyright.exe
```

Run PostgreSQL integration tests by setting a test database:

```powershell
$env:TEST_DATABASE_URL = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
.\.venv\Scripts\python.exe -m pytest
```

Validate the frontend:

```bash
cd frontend
npm ci
npm run build
```

CI validates migrations in both directions, runs backend tests, linting and
type checks, builds the frontend, and builds the production container.

## Deployment

The production image runs as a non-root user, contains the compiled React
interface, applies Alembic migrations at startup, and serves the API and UI on
port `8001`.

Before deploying:

- Use a secret manager for the Gemini key, API token, and database password.
- Terminate TLS at a reverse proxy or load balancer.
- Restrict `ALLOWED_HOSTS`, `ALLOWED_ORIGINS`, and trusted proxy addresses.
- Do not expose PostgreSQL publicly.
- Back up both the PostgreSQL and uploaded-file volumes before upgrades.

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for production configuration,
health checks, reverse-proxy guidance, backups, upgrades, and rollback.

## Project structure

```text
.
├── frontend/                 React operator interface
├── migrations/               Alembic database migrations
├── src/note_rag/
│   ├── api/                  HTTP API, middleware, and observability
│   ├── chat/                 Grounded generation and conversation handling
│   ├── chunking/             Token counting and document chunking
│   ├── context/              Reranking and context construction
│   ├── embeddings/           Embedding providers and indexing
│   ├── ingest/               Parsing, storage, pipeline, and worker
│   ├── persistence/          Database models and repositories
│   └── retrieval/            Keyword, vector, and hybrid retrieval
├── tests/                    Unit and PostgreSQL integration tests
├── docker-compose.yml        Application and pgvector stack
└── Dockerfile                Multi-stage production image
```

## Contributing

1. Create a focused branch from `main`.
2. Add or update tests with the change.
3. Run the backend checks and frontend build locally.
4. Open a pull request describing the motivation, implementation, and
   verification performed.

Bug reports and focused improvement proposals are welcome through
[GitHub Issues](https://github.com/tiongMax/note-rag/issues).

# Production deployment

Note RAG ships as one application container plus PostgreSQL with pgvector. The
application image contains the built React interface, runs Alembic migrations
at startup, serves FastAPI on port 8001, and starts the database-backed
ingestion worker.

## Required configuration

Create `.env` from `.env.example` and set at least:

```env
APP_ENVIRONMENT=production
GEMINI_API_KEY=replace-with-a-real-key
API_AUTH_TOKEN=replace-with-at-least-24-random-characters
POSTGRES_PASSWORD=replace-with-a-strong-database-password
ALLOWED_HOSTS=rag.example.com
ALLOWED_ORIGINS=https://rag.example.com
FORWARDED_ALLOW_IPS=127.0.0.1
```

Generate an API token with:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Production startup fails fast when the Gemini key or a sufficiently long API
token is missing. Process environment variables take precedence over `.env`.

## Docker Compose

Build and start the complete stack:

```powershell
docker compose up -d --build
docker compose ps
```

Open `http://127.0.0.1:8001`. The interface prompts for `API_AUTH_TOKEN` and
keeps it in browser session storage. It is not persisted after the browser
session ends.

## Guardrail controls

Production chat defaults to local input, retrieved-context, and output
guardrails. Configure `CHAT_INPUT_MAX_TOKENS`, `CHAT_PROMPT_MAX_TOKENS`,
`CHAT_PROMPT_RESERVE_TOKENS`, `CHAT_OUTPUT_HARD_MAX_TOKENS`,
`CHAT_RATE_LIMIT_REQUESTS`, and `CHAT_RATE_LIMIT_WINDOW_SECONDS` to match the
provider and deployment capacity. The groundedness and relevance thresholds
are lexical proxies controlled by `GUARDRAIL_GROUNDEDNESS_THRESHOLD` and
`GUARDRAIL_RELEVANCE_THRESHOLD`; they are not semantic judges.
`GUARDRAILS_ENABLED=false` is intended for controlled comparisons, not normal
production deployment. Production startup rejects that setting unless the
explicit emergency break-glass setting
`ALLOW_UNGUARDED_CHAT_IN_PRODUCTION=true` is also present. Treat every use as
a security incident, time-bound it, and restore guardrails immediately.

The bundled generic and chat limiters are per process and keyed by client IP.
Multi-replica deployments need an external shared limiter at the gateway to
enforce one global quota.

Inspect logs:

```powershell
docker compose logs -f app
```

Stop services without deleting data:

```powershell
docker compose down
```

Database and uploaded-file data live in the `pgvector_data` and
`note_rag_uploads` named volumes. Back up both volumes before upgrades.

## Reverse proxy

Terminate TLS at a reverse proxy or load balancer and forward traffic to port
8001. Set `ALLOWED_HOSTS` to public hostnames and `FORWARDED_ALLOW_IPS` only to
the proxy addresses whose forwarding headers should be trusted.

Do not expose PostgreSQL port 5432 publicly. The development Compose mapping on
host port 6024 should be removed or firewalled in a remote deployment.

## Operations endpoints

- `GET /health` is the process liveness probe.
- `GET /health/ready` checks database connectivity.
- `GET /metrics` exposes Prometheus-compatible process and HTTP metrics.

Every response contains `X-Request-ID`. Supplying a safe `X-Request-ID` lets a
proxy propagate its trace identifier into structured application logs.

## Security notes

- API routes require `Authorization: Bearer <API_AUTH_TOKEN>` in production.
- Request bodies and document uploads have independent byte limits.
- Per-client in-memory rate limiting protects a single application process.
  Use proxy-level distributed rate limiting when running multiple replicas.
- API documentation is disabled when `APP_ENVIRONMENT=production`.
- Rotate the API token and Gemini key through the deployment secret manager;
  never bake either value into the image.

## Upgrade and rollback

Before deploying a new image, back up PostgreSQL and uploaded files. The
container applies forward migrations automatically. To inspect migration state:

```powershell
docker compose exec app python -m alembic current
docker compose exec app python -m alembic history
```

Test migration downgrades in staging. Production rollback may require restoring
the database backup when a release includes a destructive schema change.

# Production deployment

Note RAG ships separate API, ingestion-worker, and generation-worker services,
plus PostgreSQL, Redis/Valkey, and an Nginx entry point. A one-shot migration
service applies Alembic migrations before independently deployable API and
worker replicas start.

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

Inspect logs:

```powershell
docker compose logs -f api worker generation-worker
```

Scale ingestion independently of request handling:

```powershell
docker compose up -d --scale worker=3
```

Redis consumer groups distribute each new ingestion message to one worker.
Unacknowledged messages are reclaimed after the configured worker lease, while
PostgreSQL job claims prevent concurrent duplicate processing.

Stop services without deleting data:

```powershell
docker compose down
```

Database and uploaded-file data live in the `pgvector_data` and
`note_rag_uploads` named volumes. Back up both volumes before upgrades.
Deployments spanning multiple hosts must replace or mount `note_rag_uploads`
with shared durable storage accessible to every ingestion worker.

## Reverse proxy

Terminate TLS at a reverse proxy or load balancer and forward traffic to port
8001. Set `ALLOWED_HOSTS` to public hostnames and `FORWARDED_ALLOW_IPS` only to
the proxy addresses whose forwarding headers should be trusted.

Do not expose PostgreSQL port 5432 publicly. The development Compose mapping on
host port 6024 should be removed or firewalled in a remote deployment.

## Operations endpoints

- `GET /health` is the process liveness probe.
- `GET /health/ready` checks database connectivity and Redis when stream-backed
  ingestion is enabled.
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
one-shot `migrate` service applies forward migrations. To inspect migration
state:

```powershell
docker compose run --rm migrate python -m alembic current
docker compose run --rm migrate python -m alembic history
```

Test migration downgrades in staging. Production rollback may require restoring
the database backup when a release includes a destructive schema change.

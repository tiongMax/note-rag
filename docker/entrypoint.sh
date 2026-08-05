#!/bin/sh
set -eu

python -m alembic upgrade head

exec python -m uvicorn note_rag.api.app:app \
  --host 0.0.0.0 \
  --port "${PORT:-8001}" \
  --proxy-headers \
  --forwarded-allow-ips "${FORWARDED_ALLOW_IPS:-127.0.0.1}"

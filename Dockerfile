FROM node:22-alpine AS frontend-build

WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN addgroup --system note-rag \
    && adduser --system --ingroup note-rag --home /app note-rag

COPY pyproject.toml README.md alembic.ini ./
COPY migrations/ ./migrations/
COPY src/ ./src/
RUN python -m pip install --upgrade pip \
    && python -m pip install .

COPY --from=frontend-build /build/frontend/dist ./frontend/dist
COPY docker/entrypoint.sh ./docker/entrypoint.sh

RUN sed -i 's/\r$//' ./docker/entrypoint.sh \
    && mkdir -p /app/data/uploads \
    && chown -R note-rag:note-rag /app

USER note-rag
EXPOSE 8001

HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health/ready', timeout=3)"

CMD ["sh", "/app/docker/entrypoint.sh"]

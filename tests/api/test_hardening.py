from pathlib import Path

from fastapi.testclient import TestClient
from tests.api.test_documents import (
    FakeChatProvider,
    FakeEmbeddingProvider,
)

from note_rag.api.app import create_app
from note_rag.api.settings import ApiSettings
from note_rag.ingest import LocalFileStorage
from note_rag.persistence import Database


def build_hardened_client(
    database: Database,
    tmp_path: Path,
    **overrides,
) -> TestClient:
    settings = ApiSettings(
        chunk_size=3,
        chunk_overlap=1,
        storage_path=tmp_path,
        background_worker_enabled=False,
        **overrides,
    )
    return TestClient(
        create_app(
            settings,
            database=database,
            storage=LocalFileStorage(tmp_path),
            embedding_provider=FakeEmbeddingProvider(),
            chat_provider=FakeChatProvider(),
        )
    )


def test_authentication_and_error_envelope(
    database: Database,
    tmp_path: Path,
) -> None:
    client = build_hardened_client(
        database,
        tmp_path,
        api_auth_token="a-secure-test-token",
    )

    unauthorized = client.get("/api/v1/documents")
    authorized = client.get(
        "/api/v1/documents",
        headers={"Authorization": "Bearer a-secure-test-token"},
    )

    assert unauthorized.status_code == 401
    assert unauthorized.json()["error"]["code"] == "unauthorized"
    assert unauthorized.json()["error"]["request_id"]
    assert unauthorized.headers["x-request-id"]
    assert authorized.status_code == 200


def test_security_headers_and_supplied_request_id(
    database: Database,
    tmp_path: Path,
) -> None:
    client = build_hardened_client(database, tmp_path)

    response = client.get(
        "/health",
        headers={"X-Request-ID": "test-request-123"},
    )

    assert response.headers["x-request-id"] == "test-request-123"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"


def test_request_size_limit(
    database: Database,
    tmp_path: Path,
) -> None:
    client = build_hardened_client(
        database,
        tmp_path,
        max_upload_bytes=100,
        max_request_bytes=100,
    )

    response = client.post(
        "/api/v1/chunks",
        content=b"x" * 101,
        headers={
            "Content-Type": "application/json",
            "Content-Length": "101",
        },
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


def test_rate_limit_readiness_and_metrics(
    database: Database,
    tmp_path: Path,
) -> None:
    client = build_hardened_client(
        database,
        tmp_path,
        rate_limit_requests=1,
        rate_limit_window_seconds=60,
    )

    ready = client.get("/health/ready")
    limited = client.get("/health/ready")
    metrics = client.get("/metrics")

    assert ready.status_code == 200
    assert limited.status_code == 429
    assert limited.headers["retry-after"]
    assert "note_rag_http_requests_total" in metrics.text

from pathlib import Path

from fastapi.testclient import TestClient

from note_rag.api.app import create_app
from note_rag.api.settings import ApiSettings
from note_rag.ingest import LocalFileStorage
from note_rag.persistence import Database


def build_client(
    database: Database,
    tmp_path: Path,
    *,
    max_upload_bytes: int = 1024,
) -> TestClient:
    settings = ApiSettings(
        chunk_size=3,
        chunk_overlap=1,
        storage_path=tmp_path,
        max_upload_bytes=max_upload_bytes,
    )
    return TestClient(
        create_app(
            settings,
            database=database,
            storage=LocalFileStorage(tmp_path),
        )
    )


def test_upload_and_inspect_document(
    database: Database,
    tmp_path: Path,
) -> None:
    client = build_client(database, tmp_path)

    upload = client.post(
        "/api/v1/documents",
        files={"file": ("lesson.txt", b"zero one two three four", "text/plain")},
    )

    assert upload.status_code == 201
    result = upload.json()
    assert result["status"] == "ready"
    assert result["chunk_count"] == 2

    document = client.get(f"/api/v1/documents/{result['document_id']}")
    chunks = client.get(f"/api/v1/documents/{result['document_id']}/chunks")
    job = client.get(f"/api/v1/ingestion-jobs/{result['job_id']}")

    assert document.status_code == 200
    assert document.json()["filename"] == "lesson.txt"
    assert [item["text"] for item in chunks.json()] == [
        "zero one two",
        "two three four",
    ]
    assert job.json()["status"] == "completed"
    assert job.json()["progress"] == 100


def test_duplicate_upload_returns_existing_document(
    database: Database,
    tmp_path: Path,
) -> None:
    client = build_client(database, tmp_path)
    files = {"file": ("lesson.md", b"# Same content", "text/markdown")}

    first = client.post("/api/v1/documents", files=files)
    duplicate = client.post("/api/v1/documents", files=files)

    assert first.status_code == 201
    assert duplicate.status_code == 200
    assert duplicate.json()["duplicate"] is True
    assert duplicate.json()["document_id"] == first.json()["document_id"]


def test_rejects_unsupported_and_oversized_uploads(
    database: Database,
    tmp_path: Path,
) -> None:
    client = build_client(database, tmp_path, max_upload_bytes=4)

    unsupported = client.post(
        "/api/v1/documents",
        files={"file": ("notes.docx", b"text", "application/octet-stream")},
    )
    oversized = client.post(
        "/api/v1/documents",
        files={"file": ("notes.txt", b"12345", "text/plain")},
    )

    assert unsupported.status_code == 415
    assert oversized.status_code == 413


def test_failed_parse_is_inspectable(
    database: Database,
    tmp_path: Path,
) -> None:
    client = build_client(database, tmp_path)

    upload = client.post(
        "/api/v1/documents",
        files={"file": ("broken.txt", b"\xff\xfe", "text/plain")},
    )

    assert upload.status_code == 422
    result = upload.json()
    assert result["status"] == "failed"
    document = client.get(f"/api/v1/documents/{result['document_id']}")
    assert document.status_code == 200
    assert document.json()["status"] == "failed"

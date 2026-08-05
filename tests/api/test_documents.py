import time
from collections.abc import Iterator
from pathlib import Path

from fastapi.testclient import TestClient

from note_rag.api.app import create_app
from note_rag.api.settings import ApiSettings
from note_rag.chat import ChatTurn
from note_rag.ingest import LocalFileStorage
from note_rag.persistence import Database


class FakeEmbeddingProvider:
    model_name = "fake-768"
    dimension = 768

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, *([0.0] * 767)] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [1.0, *([0.0] * 767)]


class FakeChatProvider:
    model_name = "fake-chat"

    def generate(
        self,
        system_instruction: str,
        turns: list[ChatTurn],
    ) -> str:
        return "Apples grow in orchards [1]."

    def stream(
        self,
        system_instruction: str,
        turns: list[ChatTurn],
    ) -> Iterator[str]:
        yield "Apples grow "
        yield "in orchards [1]."


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
        background_worker_enabled=False,
        worker_max_attempts=1,
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


def build_background_client(database: Database, tmp_path: Path) -> TestClient:
    settings = ApiSettings(
        chunk_size=3,
        chunk_overlap=1,
        storage_path=tmp_path,
        background_worker_enabled=True,
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


def test_upload_queues_background_job(
    database: Database,
    tmp_path: Path,
) -> None:
    client = build_background_client(database, tmp_path)

    upload = client.post(
        "/api/v1/documents",
        files={
            "file": (
                "background.txt",
                b"queued for background processing",
                "text/plain",
            )
        },
    )
    job = client.get(
        f"/api/v1/ingestion-jobs/{upload.json()['job_id']}"
    )

    assert upload.status_code == 202
    assert upload.json()["status"] == "pending"
    assert job.status_code == 200
    assert job.json()["status"] == "queued"
    assert job.json()["attempts"] == 0


def test_lifespan_worker_completes_queued_upload(
    database: Database,
    tmp_path: Path,
) -> None:
    settings = ApiSettings(
        chunk_size=3,
        chunk_overlap=1,
        storage_path=tmp_path,
        background_worker_enabled=True,
        worker_poll_interval_seconds=0.01,
        worker_retry_backoff_seconds=0,
    )
    app = create_app(
        settings,
        database=database,
        storage=LocalFileStorage(tmp_path),
        embedding_provider=FakeEmbeddingProvider(),
        chat_provider=FakeChatProvider(),
    )

    with TestClient(app) as client:
        upload = client.post(
            "/api/v1/documents",
            files={
                "file": (
                    "lifespan.txt",
                    b"processed by lifespan worker",
                    "text/plain",
                )
            },
        )
        deadline = time.monotonic() + 2
        job = client.get(
            f"/api/v1/ingestion-jobs/{upload.json()['job_id']}"
        )
        while time.monotonic() < deadline:
            if job.json()["status"] == "completed":
                break
            time.sleep(0.01)
            job = client.get(
                f"/api/v1/ingestion-jobs/{upload.json()['job_id']}"
            )

    assert upload.status_code == 202
    assert job.json()["status"] == "completed"
    assert job.json()["progress"] == 100
    assert job.json()["worker_id"] is None


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
    assert result["indexing_status"] == "indexed"
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

    reindex = client.post(
        f"/api/v1/documents/{result['document_id']}/index"
    )
    assert reindex.status_code == 200
    assert reindex.json()["status"] == "indexed"


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


def test_searches_uploaded_chunks(database: Database, tmp_path: Path) -> None:
    client = build_client(database, tmp_path)
    upload = client.post(
        "/api/v1/documents",
        files={
            "file": (
                "retrieval.txt",
                b"apples grow in orchards",
                "text/plain",
            )
        },
    )

    search = client.post(
        "/api/v1/retrieval/search",
        json={
            "query": "apples",
            "mode": "hybrid",
            "filters": {"filenames": ["retrieval.txt"]},
        },
    )

    assert upload.status_code == 201
    assert search.status_code == 200
    assert search.json()["hits"][0]["filename"] == "retrieval.txt"
    assert search.json()["hits"][0]["keyword_score"] is not None


def test_builds_reranked_context(database: Database, tmp_path: Path) -> None:
    client = build_client(database, tmp_path)
    upload = client.post(
        "/api/v1/documents",
        files={
            "file": (
                "context.txt",
                b"apples grow in orchards with careful cultivation",
                "text/plain",
            )
        },
    )

    context = client.post(
        "/api/v1/retrieval/context",
        json={
            "query": "apple orchards",
            "max_context_tokens": 80,
            "filters": {"filenames": ["context.txt"]},
        },
    )

    assert upload.status_code == 201
    assert context.status_code == 200
    body = context.json()
    assert body["chunks"][0]["filename"] == "context.txt"
    assert body["chunks"][0]["citation_id"] == 1
    assert body["chunks"][0]["source_metadata"]["source_id"] == "context.txt"
    assert body["reranker_model"] == "lexical-overlap-v1"
    assert body["token_count"] <= body["token_budget"]


def test_chat_persists_history_and_citations(
    database: Database,
    tmp_path: Path,
) -> None:
    client = build_client(database, tmp_path)
    upload = client.post(
        "/api/v1/documents",
        files={
            "file": (
                "chat.txt",
                b"apples grow in orchards",
                "text/plain",
            )
        },
    )
    first = client.post(
        "/api/v1/chat",
        json={
            "query": "Where do apples grow?",
            "filters": {"filenames": ["chat.txt"]},
        },
    )
    follow_up = client.post(
        "/api/v1/chat",
        json={
            "query": "Can you repeat that?",
            "conversation_id": first.json()["conversation_id"],
            "filters": {"filenames": ["chat.txt"]},
        },
    )
    conversation = client.get(
        f"/api/v1/conversations/{first.json()['conversation_id']}"
    )

    assert upload.status_code == 201
    assert first.status_code == 200
    assert first.json()["citations"][0]["filename"] == "chat.txt"
    assert follow_up.status_code == 200
    assert conversation.status_code == 200
    assert conversation.json()["message_count"] == 4
    assert [item["role"] for item in conversation.json()["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_streams_chat_events(database: Database, tmp_path: Path) -> None:
    client = build_client(database, tmp_path)
    client.post(
        "/api/v1/documents",
        files={
            "file": (
                "stream.txt",
                b"apples grow in orchards",
                "text/plain",
            )
        },
    )

    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={
            "query": "Where do apples grow?",
            "filters": {"filenames": ["stream.txt"]},
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: metadata" in body
    assert "event: delta" in body
    assert "event: done" in body
    assert "Apples grow" in body

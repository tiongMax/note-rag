from datetime import UTC, datetime, timedelta
from pathlib import Path

from note_rag.embeddings import IndexingService
from note_rag.ingest import IngestionPipeline, IngestionWorker, LocalFileStorage
from note_rag.persistence import (
    ChunkRepository,
    Database,
    DocumentRepository,
    DocumentStatus,
    IndexingStatus,
    IngestionJobRepository,
    IngestionJobStatus,
)


class FakeEmbeddingProvider:
    model_name = "fake-768"
    dimension = 768

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, *([0.0] * 767)] for _ in texts]


class FlakyEmbeddingProvider(FakeEmbeddingProvider):
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("temporary embedding failure")
        return super().embed(texts)


def build_worker(
    database: Database,
    tmp_path: Path,
    provider,
    *,
    max_attempts: int = 3,
    lease_timeout_seconds: float = 300,
) -> tuple[IngestionPipeline, IngestionWorker]:
    pipeline = IngestionPipeline(database, LocalFileStorage(tmp_path))
    worker = IngestionWorker(
        database,
        pipeline,
        IndexingService(database, provider),
        max_attempts=max_attempts,
        retry_backoff_seconds=0,
        poll_interval_seconds=0.01,
        lease_timeout_seconds=lease_timeout_seconds,
        worker_id="test-worker",
    )
    return pipeline, worker


def test_processes_queued_job_through_indexing(
    database: Database,
    tmp_path: Path,
) -> None:
    pipeline, worker = build_worker(
        database,
        tmp_path,
        FakeEmbeddingProvider(),
    )
    queued = pipeline.enqueue(
        filename="queued.txt",
        media_type="text/plain",
        content=b"queued background ingestion",
    )
    assert queued.job_id is not None

    assert queued.status is DocumentStatus.PENDING
    assert worker.run_once() is True
    assert worker.run_once() is False
    with database.session() as session:
        document = DocumentRepository(session).get(queued.document_id)
        job = IngestionJobRepository(session).get(queued.job_id)
        assert document is not None
        assert job is not None
        assert document.status is DocumentStatus.READY
        assert document.indexing_status is IndexingStatus.INDEXED
        assert len(ChunkRepository(session).list_for_document(document.id)) == 1
        assert job.status is IngestionJobStatus.COMPLETED
        assert job.progress == 100
        assert job.attempts == 1
        assert job.worker_id is None
        assert job.locked_at is None


def test_retries_indexing_without_recreating_chunks(
    database: Database,
    tmp_path: Path,
) -> None:
    provider = FlakyEmbeddingProvider(failures=1)
    pipeline, worker = build_worker(database, tmp_path, provider)
    queued = pipeline.enqueue(
        filename="retry.txt",
        media_type="text/plain",
        content=b"retry the embedding stage",
    )
    assert queued.job_id is not None

    assert worker.run_once() is True
    with database.session() as session:
        job = IngestionJobRepository(session).get(queued.job_id)
        chunks = ChunkRepository(session).list_for_document(queued.document_id)
        assert job is not None
        assert job.status is IngestionJobStatus.QUEUED
        assert job.attempts == 1
        assert job.error_message == "temporary embedding failure"
        assert len(chunks) == 1

    assert worker.run_once() is True
    with database.session() as session:
        job = IngestionJobRepository(session).get(queued.job_id)
        chunks = ChunkRepository(session).list_for_document(queued.document_id)
        assert job is not None
        assert job.status is IngestionJobStatus.COMPLETED
        assert job.attempts == 2
        assert len(chunks) == 1


def test_persists_terminal_parse_failure(
    database: Database,
    tmp_path: Path,
) -> None:
    pipeline, worker = build_worker(
        database,
        tmp_path,
        FakeEmbeddingProvider(),
        max_attempts=2,
    )
    queued = pipeline.enqueue(
        filename="broken.txt",
        media_type="text/plain",
        content=b"\xff\xfe",
    )
    assert queued.job_id is not None

    assert worker.run_once() is True
    assert worker.run_once() is True
    with database.session() as session:
        job = IngestionJobRepository(session).get(queued.job_id)
        document = DocumentRepository(session).get(queued.document_id)
        assert job is not None
        assert document is not None
        assert job.status is IngestionJobStatus.FAILED
        assert job.attempts == 2
        assert job.finished_at is not None
        assert "valid UTF-8" in (job.error_message or "")
        assert document.status is DocumentStatus.FAILED


def test_recovers_stale_worker_lease(
    database: Database,
    tmp_path: Path,
) -> None:
    pipeline, worker = build_worker(
        database,
        tmp_path,
        FakeEmbeddingProvider(),
        lease_timeout_seconds=1,
    )
    queued = pipeline.enqueue(
        filename="recover.txt",
        media_type="text/plain",
        content=b"restart recovery",
    )
    assert queued.job_id is not None
    with database.session() as session:
        claimed = IngestionJobRepository(session).claim(
            queued.job_id,
            worker_id="dead-worker",
            now=datetime.now(UTC) - timedelta(minutes=5),
        )
        assert claimed is not None

    assert worker.recover_stale_jobs() == 1
    with database.session() as session:
        job = IngestionJobRepository(session).get(queued.job_id)
        assert job is not None
        assert job.status is IngestionJobStatus.QUEUED
        assert job.worker_id is None
        assert job.error_message == "recovered stale worker lease"

    assert worker.run_job(queued.job_id) is True

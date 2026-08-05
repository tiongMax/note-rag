import os
import uuid
from pathlib import Path

import pytest

from note_rag.embeddings import IndexingService
from note_rag.ingest import IngestionPipeline, IngestionWorker, LocalFileStorage
from note_rag.persistence import (
    Database,
    DocumentRepository,
    IndexingStatus,
    IngestionJobRepository,
    IngestionJobStatus,
)
from note_rag.persistence.settings import DatabaseSettings

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")


class FakeEmbeddingProvider:
    model_name = "fake-768"
    dimension = 768

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, *([0.0] * 767)] for _ in texts]


@pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL is required for the PostgreSQL integration test",
)
def test_live_background_worker_round_trip(tmp_path: Path) -> None:
    assert TEST_DATABASE_URL is not None
    database = Database(DatabaseSettings(url=TEST_DATABASE_URL))
    pipeline = IngestionPipeline(database, LocalFileStorage(tmp_path))
    worker = IngestionWorker(
        database,
        pipeline,
        IndexingService(database, FakeEmbeddingProvider()),
        retry_backoff_seconds=0,
        worker_id="postgres-smoke",
    )
    queued = pipeline.enqueue(
        filename="phase-eight-smoke.txt",
        media_type="text/plain",
        content=f"background worker smoke {uuid.uuid4()}".encode(),
    )
    try:
        assert queued.job_id is not None
        assert worker.run_job(queued.job_id) is True
        with database.session() as session:
            document = DocumentRepository(session).get(queued.document_id)
            job = IngestionJobRepository(session).get(queued.job_id)
            assert document is not None
            assert job is not None
            assert document.indexing_status is IndexingStatus.INDEXED
            assert job.status is IngestionJobStatus.COMPLETED
            assert job.attempts == 1
            assert job.worker_id is None
    finally:
        with database.session() as session:
            document = DocumentRepository(session).get(queued.document_id)
            if document is not None:
                DocumentRepository(session).delete(document)
        database.dispose()

"""Tests for the new Redis-backed IngestionWorker."""

import uuid
from unittest.mock import Mock, patch

import pytest

from note_rag.ingest.pipeline import IngestionResult
from note_rag.ingest.worker import IngestionWorker
from note_rag.persistence import DocumentStatus, IngestionJobStatus
from note_rag.queue.redis_stream import RedisMsg


@pytest.fixture
def mock_pipeline():
    pipeline = Mock()
    pipeline.process_job.return_value = IngestionResult(
        document_id=uuid.uuid4(),
        job_id=uuid.uuid4(),
        status=DocumentStatus.READY,
        duplicate=False,
        chunk_count=10,
        token_count=100,
    )
    return pipeline


@pytest.fixture
def mock_indexing():
    indexing = Mock()
    mock_result = Mock()
    mock_result.error_message = None
    indexing.index_document.return_value = mock_result
    return indexing


@pytest.fixture
def mock_consumer():
    return Mock()


@pytest.fixture
def worker(mock_pipeline, mock_indexing, mock_consumer):
    db = Mock()
    db.session.return_value.__enter__ = Mock(return_value=Mock())
    db.session.return_value.__exit__ = Mock(return_value=None)

    return IngestionWorker(
        database=db,
        pipeline=mock_pipeline,
        indexing_service=mock_indexing,
        consumer=mock_consumer,
        max_attempts=3,
    )


@patch("note_rag.ingest.worker.IngestionJobRepository")
def test_worker_run_once_empty(repository_class, worker, mock_consumer):
    mock_consumer.poll.return_value = None
    repository_class.return_value.claim_next.return_value = None
    assert worker.run_once() is False


@patch("note_rag.ingest.worker.IngestionJobRepository")
def test_worker_uses_database_fallback_when_stream_is_empty(
    repository_class,
    worker,
    mock_consumer,
    mock_pipeline,
):
    job_id = uuid.uuid4()
    repository = repository_class.return_value
    repository.claim_next.return_value = Mock(id=job_id)
    repository.get.return_value = Mock()
    mock_consumer.poll.return_value = None

    assert worker.run_once() is True

    mock_pipeline.process_job.assert_called_once_with(job_id)
    mock_consumer.ack.assert_not_called()


@patch("note_rag.ingest.worker.IngestionJobRepository")
def test_worker_success_acks_message(
    repository_class, worker, mock_consumer, mock_pipeline
):
    job_id = uuid.uuid4()
    msg = RedisMsg("s", "g", "c", b"1", {"job_id": str(job_id)})
    mock_consumer.poll.return_value = (job_id, msg)

    repository = repository_class.return_value
    repository.claim.return_value = Mock()
    repository.get.return_value = Mock()

    assert worker.run_once() is True

    mock_pipeline.process_job.assert_called_once_with(job_id)
    mock_consumer.ack.assert_called_once_with(msg)


@patch("note_rag.ingest.worker.DocumentRepository")
@patch("note_rag.ingest.worker.IngestionJobRepository")
def test_worker_reschedules_and_defers_retry(
    repository_class,
    document_repository_class,
    worker,
    mock_consumer,
    mock_pipeline,
):
    job_id = uuid.uuid4()
    document_id = uuid.uuid4()
    msg = RedisMsg("s", "g", "c", b"2", {"job_id": str(job_id)})
    job = Mock(attempts=1, document_id=document_id)
    repository = repository_class.return_value
    repository.claim.return_value = job
    repository.get.return_value = job
    document_repository_class.return_value.get.return_value = Mock()
    mock_pipeline.process_job.side_effect = RuntimeError("temporary failure")
    mock_consumer.poll.return_value = (job_id, msg)

    assert worker.run_once() is True

    repository.reschedule.assert_called_once()
    mock_consumer.defer.assert_called_once_with(msg, delay_seconds=2.0)
    mock_consumer.ack.assert_not_called()


@patch("note_rag.ingest.worker.IngestionJobRepository")
def test_worker_acks_terminal_duplicate(
    repository_class, worker, mock_consumer, mock_pipeline
):
    job_id = uuid.uuid4()
    msg = RedisMsg("s", "g", "c", b"3", {"job_id": str(job_id)})
    repository = repository_class.return_value
    repository.claim.return_value = None
    repository.get.return_value = Mock(status=IngestionJobStatus.COMPLETED)
    mock_consumer.poll.return_value = (job_id, msg)

    assert worker.run_once() is True

    mock_pipeline.process_job.assert_not_called()
    mock_consumer.ack.assert_called_once_with(msg)

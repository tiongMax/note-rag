"""Redis Stream-backed ingestion worker — replaces the DB-polling approach."""

from __future__ import annotations

import logging
import threading
import uuid

from note_rag.embeddings import IndexingService
from note_rag.ingest.pipeline import IngestionPipeline
from note_rag.persistence import (
    Database,
    DocumentRepository,
    DocumentStatus,
    IndexingStatus,
    IngestionJobRepository,
)
from note_rag.queue import QueueConsumer
from note_rag.queue.redis_stream import RedisMsg

logger = logging.getLogger(__name__)


class IngestionWorker:
    """Consumes job IDs from a Redis Stream and runs the ingestion pipeline.

    Key differences from the old DB-polling worker:
    - ``run_once()`` calls ``QueueConsumer.poll()`` (blocks ≤5 s) instead of
      ``IngestionJobRepository.claim_next()`` with SKIP LOCKED.
    - Messages are **acked** only on success; on failure they remain in the
      pending-entries list and are re-delivered after ``min_idle_ms``.
    - ``run_forever()`` calls ``consumer.start()`` on entry to recover any
      messages that were in-flight when the previous instance crashed.
    """

    def __init__(
        self,
        database: Database,
        pipeline: IngestionPipeline,
        indexing_service: IndexingService,
        consumer: QueueConsumer,
        *,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 2.0,
        worker_id: str | None = None,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be greater than zero")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds cannot be negative")
        self.database = database
        self.pipeline = pipeline
        self.indexing_service = indexing_service
        self.consumer = consumer
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.worker_id = worker_id or uuid.uuid4().hex

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_once(self) -> bool:
        """Poll for one message and process it.  Returns True if a job ran."""
        result = self.consumer.poll()
        if result is None:
            return False
        job_id, msg = result
        self._execute(job_id, msg)
        return True

    def run_forever(self, stop_event: threading.Event) -> None:
        """Block-poll loop.  Call this from a dedicated worker process/thread."""
        self.consumer.start()  # ensure group exists + recover pending
        while not stop_event.is_set():
            try:
                self.run_once()
            except Exception:
                logger.exception("Ingestion worker iteration failed")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _execute(self, job_id: uuid.UUID, msg: RedisMsg) -> None:
        try:
            ingestion = self.pipeline.process_job(job_id)
        except Exception as error:
            self._handle_failure(
                job_id,
                msg,
                str(error) or error.__class__.__name__,
                indexing_failure=False,
            )
            return

        try:
            indexing = self.indexing_service.index_document(
                ingestion.document_id,
                job_id=job_id,
            )
        except Exception as error:
            self._handle_failure(
                job_id,
                msg,
                str(error) or error.__class__.__name__,
                indexing_failure=True,
            )
            return

        if indexing.error_message is not None:
            self._handle_failure(
                job_id,
                msg,
                indexing.error_message,
                indexing_failure=True,
            )
            return

        # Success — mark job complete in DB and ack the Redis message.
        with self.database.session() as session:
            job = IngestionJobRepository(session).get(job_id)
            if job is None:
                raise LookupError("ingestion job not found")
            IngestionJobRepository(session).complete_claim(job)
        self.consumer.ack(msg)

    def _handle_failure(
        self,
        job_id: uuid.UUID,
        msg: RedisMsg,
        error_message: str,
        *,
        indexing_failure: bool,
    ) -> None:
        with self.database.session() as session:
            jobs = IngestionJobRepository(session)
            job = jobs.get(job_id)
            if job is None:
                raise LookupError("ingestion job not found")
            document = DocumentRepository(session).get(job.document_id)
            if document is None:
                raise LookupError("document not found")

            if job.attempts < self.max_attempts:
                # Don't ack — Redis will re-deliver after the idle timeout.
                logger.warning(
                    "Job %s failed (attempt %d/%d): %s — will retry via Redis",
                    job_id,
                    job.attempts,
                    self.max_attempts,
                    error_message,
                )
                if not indexing_failure:
                    document.status = DocumentStatus.PENDING
                    document.error_message = None
                return

            # Max attempts exhausted — ack to stop re-delivery and mark failed.
            jobs.fail(job, error_message=error_message)
            if indexing_failure:
                document.indexing_status = IndexingStatus.FAILED
                document.indexing_error = error_message
            else:
                document.status = DocumentStatus.FAILED
                document.error_message = error_message
        self.consumer.ack(msg)

"""Redis Stream-backed ingestion worker — replaces the DB-polling approach."""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import UTC, datetime, timedelta

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
        consumer: QueueConsumer | None = None,
        *,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 2.0,
        poll_interval_seconds: float = 1.0,
        lease_timeout_seconds: float = 300.0,
        worker_id: str | None = None,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be greater than zero")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds cannot be negative")
        if lease_timeout_seconds <= 0:
            raise ValueError("lease_timeout_seconds must be greater than zero")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be greater than zero")
        self.database = database
        self.pipeline = pipeline
        self.indexing_service = indexing_service
        self.consumer = consumer
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.lease_timeout_seconds = lease_timeout_seconds
        self.worker_id = worker_id or uuid.uuid4().hex

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_once(self) -> bool:
        """Poll for one message and process it.  Returns True if a job ran."""
        if self.consumer is not None:
            try:
                result = self.consumer.poll()
            except Exception:
                logger.exception(
                    "Redis ingestion poll failed; using database fallback"
                )
                result = None
            if result is not None:
                job_id, msg = result
                self._execute(job_id, msg)
                return True

        now = datetime.now(UTC)
        with self.database.session() as session:
            job = IngestionJobRepository(session).claim_next(
                worker_id=self.worker_id,
                now=now,
            )
            job_id = job.id if job is not None else None
        if job_id is None:
            return False
        self._execute(job_id, already_claimed=True)
        return True

    def run_job(self, job_id: uuid.UUID) -> bool:
        """Claim and synchronously process one job without a queue message."""
        return self._execute(job_id)

    def run_forever(self, stop_event: threading.Event) -> None:
        """Block-poll loop.  Call this from a dedicated worker process/thread."""
        self.recover_stale_jobs()
        queue_started = self.consumer is None
        while not stop_event.is_set():
            if not queue_started and self.consumer is not None:
                try:
                    self.consumer.start()
                    queue_started = True
                except Exception:
                    logger.exception(
                        "Redis ingestion startup failed; using database fallback"
                    )
            try:
                if self.run_once():
                    continue
            except Exception:
                logger.exception("Ingestion worker iteration failed")
            stop_event.wait(self.poll_interval_seconds)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _execute(
        self,
        job_id: uuid.UUID,
        msg: RedisMsg | None = None,
        *,
        already_claimed: bool = False,
    ) -> bool:
        if not already_claimed:
            now = datetime.now(UTC)
            with self.database.session() as session:
                jobs = IngestionJobRepository(session)
                job = jobs.claim(job_id, worker_id=self.worker_id, now=now)
                if job is None:
                    current = jobs.get(job_id)
                    if msg is not None and self.consumer is not None:
                        if current is None or current.status.value in {
                            "completed",
                            "failed",
                        }:
                            self.consumer.ack(msg)
                        else:
                            self.consumer.defer(
                                msg,
                                delay_seconds=self.retry_backoff_seconds,
                            )
                    return False

        try:
            ingestion = self.pipeline.process_job(job_id)
        except Exception as error:
            self._handle_failure(
                job_id,
                msg,
                str(error) or error.__class__.__name__,
                indexing_failure=False,
            )
            return True

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
            return True

        if indexing.error_message is not None:
            self._handle_failure(
                job_id,
                msg,
                indexing.error_message,
                indexing_failure=True,
            )
            return True

        # Success — mark job complete in DB and ack the Redis message.
        with self.database.session() as session:
            job = IngestionJobRepository(session).get(job_id)
            if job is None:
                raise LookupError("ingestion job not found")
            IngestionJobRepository(session).complete_claim(job)
        if msg is not None and self.consumer is not None:
            self.consumer.ack(msg)
        return True

    def _handle_failure(
        self,
        job_id: uuid.UUID,
        msg: RedisMsg | None,
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
                delay = self.retry_backoff_seconds * (2 ** (job.attempts - 1))
                jobs.reschedule(
                    job,
                    error_message=error_message,
                    next_attempt_at=datetime.now(UTC) + timedelta(seconds=delay),
                )
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
                if msg is not None and self.consumer is not None:
                    self.consumer.defer(msg, delay_seconds=delay)
                return

            # Max attempts exhausted — ack to stop re-delivery and mark failed.
            jobs.fail(job, error_message=error_message)
            if indexing_failure:
                document.indexing_status = IndexingStatus.FAILED
                document.indexing_error = error_message
            else:
                document.status = DocumentStatus.FAILED
                document.error_message = error_message
        if msg is not None and self.consumer is not None:
            self.consumer.ack(msg)

    def recover_stale_jobs(self) -> int:
        now = datetime.now(UTC)
        stale_before = now - timedelta(seconds=self.lease_timeout_seconds)
        with self.database.session() as session:
            return IngestionJobRepository(session).recover_stale(
                stale_before=stale_before,
                now=now,
            )

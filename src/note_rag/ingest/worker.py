"""Database-backed ingestion worker with leases, retries, and recovery."""

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

logger = logging.getLogger(__name__)


class IngestionWorker:
    def __init__(
        self,
        database: Database,
        pipeline: IngestionPipeline,
        indexing_service: IndexingService,
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
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be greater than zero")
        if lease_timeout_seconds <= 0:
            raise ValueError("lease_timeout_seconds must be greater than zero")
        self.database = database
        self.pipeline = pipeline
        self.indexing_service = indexing_service
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.lease_timeout_seconds = lease_timeout_seconds
        self.worker_id = worker_id or uuid.uuid4().hex

    def run_once(self) -> bool:
        now = datetime.now(UTC)
        with self.database.session() as session:
            job = IngestionJobRepository(session).claim_next(
                worker_id=self.worker_id,
                now=now,
            )
            job_id = job.id if job is not None else None
        if job_id is None:
            return False
        self._execute(job_id)
        return True

    def run_job(self, job_id: uuid.UUID) -> bool:
        now = datetime.now(UTC)
        with self.database.session() as session:
            job = IngestionJobRepository(session).claim(
                job_id,
                worker_id=self.worker_id,
                now=now,
            )
        if job is None:
            return False
        self._execute(job_id)
        return True

    def recover_stale_jobs(self) -> int:
        now = datetime.now(UTC)
        stale_before = now - timedelta(seconds=self.lease_timeout_seconds)
        with self.database.session() as session:
            return IngestionJobRepository(session).recover_stale(
                stale_before=stale_before,
                now=now,
            )

    def run_forever(self, stop_event: threading.Event) -> None:
        self.recover_stale_jobs()
        while not stop_event.is_set():
            try:
                if self.run_once():
                    continue
            except Exception:
                logger.exception("background ingestion worker iteration failed")
            stop_event.wait(self.poll_interval_seconds)

    def _execute(self, job_id: uuid.UUID) -> None:
        try:
            ingestion = self.pipeline.process_job(job_id)
        except Exception as error:
            self._handle_failure(
                job_id,
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
                str(error) or error.__class__.__name__,
                indexing_failure=True,
            )
            return
        if indexing.error_message is not None:
            self._handle_failure(
                job_id,
                indexing.error_message,
                indexing_failure=True,
            )
            return

        with self.database.session() as session:
            job = IngestionJobRepository(session).get(job_id)
            if job is None:
                raise LookupError("ingestion job not found")
            IngestionJobRepository(session).complete_claim(job)

    def _handle_failure(
        self,
        job_id: uuid.UUID,
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
                if not indexing_failure:
                    document.status = DocumentStatus.PENDING
                    document.error_message = None
                return

            jobs.fail(job, error_message=error_message)
            if indexing_failure:
                document.indexing_status = IndexingStatus.FAILED
                document.indexing_error = error_message
            else:
                document.status = DocumentStatus.FAILED
                document.error_message = error_message

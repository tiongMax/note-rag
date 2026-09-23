"""Queueable parse-and-chunk ingestion pipeline."""

import hashlib
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from note_rag.chunking import Chunker, TokenChunker
from note_rag.ingest.parsers import ParserRegistry
from note_rag.ingest.storage import LocalFileStorage
from note_rag.persistence import (
    ChunkRepository,
    Database,
    Document,
    DocumentRepository,
    DocumentStatus,
    IngestionJob,
    IngestionJobRepository,
    IngestionJobStatus,
)

if TYPE_CHECKING:
    from note_rag.queue import QueuePublisher

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestionResult:
    document_id: uuid.UUID
    job_id: uuid.UUID | None
    status: DocumentStatus
    duplicate: bool
    chunk_count: int
    token_count: int
    error_message: str | None = None


class IngestionPipeline:
    def __init__(
        self,
        database: Database,
        storage: LocalFileStorage,
        *,
        parser_registry: ParserRegistry | None = None,
        chunker: Chunker | None = None,
        queue_publisher: "QueuePublisher | None" = None,
    ) -> None:
        self.database = database
        self.storage = storage
        self.parser_registry = parser_registry or ParserRegistry()
        self.chunker = chunker or TokenChunker()
        self.queue_publisher = queue_publisher

    def ingest(
        self,
        *,
        filename: str,
        media_type: str,
        content: bytes,
    ) -> IngestionResult:
        """Compatibility wrapper that processes one upload synchronously."""

        queued = self.enqueue(
            filename=filename,
            media_type=media_type,
            content=content,
        )
        if queued.duplicate or queued.job_id is None:
            return queued
        with self.database.session() as session:
            job = IngestionJobRepository(session).get(queued.job_id)
            if job is None:
                raise LookupError("ingestion job not found")
            job.attempts += 1
            job.started_at = datetime.now(UTC)
        try:
            return self.process_job(queued.job_id, complete_job=True)
        except Exception as error:
            message = str(error) or error.__class__.__name__
            with self.database.session() as session:
                job = IngestionJobRepository(session).get(queued.job_id)
                document = DocumentRepository(session).get(queued.document_id)
                if job is None or document is None:
                    raise LookupError("queued ingestion state was not found") from error
                document.status = DocumentStatus.FAILED
                document.error_message = message
                IngestionJobRepository(session).fail(
                    job,
                    error_message=message,
                )
                return self._result(
                    document,
                    job_id=job.id,
                    duplicate=False,
                )

    def enqueue(
        self,
        *,
        filename: str,
        media_type: str,
        content: bytes,
    ) -> IngestionResult:
        content_hash = hashlib.sha256(content).hexdigest()
        with self.database.session() as session:
            documents = DocumentRepository(session)
            existing = documents.get_by_content_hash(content_hash)
            if existing is not None:
                jobs = IngestionJobRepository(session).list_for_document(existing.id)
                return self._result(
                    existing,
                    job_id=jobs[-1].id if jobs else None,
                    duplicate=True,
                )

            document = documents.add(
                Document(
                    filename=filename,
                    media_type=media_type,
                    content_hash=content_hash,
                )
            )
            stored_file = self.storage.save(
                content,
                filename=filename,
                content_hash=content_hash,
            )
            document.storage_uri = stored_file.uri
            job_repository = IngestionJobRepository(session)
            job = job_repository.add(
                IngestionJob(
                    document=document,
                    attempts=0,
                )
            )
            result = self._result(document, job_id=job.id, duplicate=False)

        # Publish to Redis Stream *after* the DB transaction is committed so
        # the worker can always find the row when it picks up the message.
        if self.queue_publisher is not None and result.job_id is not None:
            try:
                self.queue_publisher.enqueue(result.job_id)
            except Exception:
                # The durable PostgreSQL row remains the source of truth. Redis
                # workers fall back to claiming queued rows, so a transient
                # publish outage cannot strand an accepted upload.
                logger.exception(
                    "failed to publish ingestion job %s; workers will use the "
                    "database fallback",
                    result.job_id,
                )
        return result

    def process_job(
        self,
        job_id: uuid.UUID,
        *,
        complete_job: bool = False,
    ) -> IngestionResult:
        with self.database.session() as session:
            jobs = IngestionJobRepository(session)
            job = jobs.get(job_id)
            if job is None:
                raise LookupError("ingestion job not found")
            document = DocumentRepository(session).get(job.document_id)
            if document is None:
                raise LookupError("document not found")

            chunks = ChunkRepository(session).list_for_document(document.id)
            if not chunks:
                jobs.set_status(
                    job,
                    IngestionJobStatus.PARSING,
                    progress=10,
                )
                if document.storage_uri is None:
                    raise RuntimeError("document has no stored source file")
                content = self.storage.read(document.storage_uri)
                parsed = self.parser_registry.get(document.filename).parse(content)
                jobs.set_status(
                    job,
                    IngestionJobStatus.CHUNKING,
                    progress=40,
                )
                generated = self.chunker.chunk(
                    parsed.text,
                    source_id=document.filename,
                )
                ChunkRepository(session).add_from_chunks(
                    document,
                    generated,
                    metadata_for_chunk=lambda chunk: parsed.metadata_for_range(
                        chunk.metadata.char_start,
                        chunk.metadata.char_end,
                    ),
                )
                document.status = DocumentStatus.READY
                document.error_message = None

            if complete_job:
                jobs.complete_claim(job)
            else:
                jobs.set_status(
                    job,
                    IngestionJobStatus.EMBEDDING,
                    progress=60,
                )
            return self._result(document, job_id=job.id, duplicate=False)

    @staticmethod
    def _result(
        document: Document,
        *,
        job_id: uuid.UUID | None,
        duplicate: bool,
    ) -> IngestionResult:
        return IngestionResult(
            document_id=document.id,
            job_id=job_id,
            status=document.status,
            duplicate=duplicate,
            chunk_count=document.chunk_count,
            token_count=document.token_count,
            error_message=document.error_message,
        )

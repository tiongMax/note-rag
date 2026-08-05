"""Synchronous parse-and-chunk ingestion pipeline."""

import hashlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from note_rag.chunking import TokenChunker
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
        chunker: TokenChunker | None = None,
    ) -> None:
        self.database = database
        self.storage = storage
        self.parser_registry = parser_registry or ParserRegistry()
        self.chunker = chunker or TokenChunker()

    def ingest(
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
            job_repository = IngestionJobRepository(session)
            job = job_repository.add(
                IngestionJob(
                    document=document,
                    attempts=1,
                    started_at=datetime.now(UTC),
                )
            )

            try:
                job_repository.set_status(
                    job,
                    IngestionJobStatus.PARSING,
                    progress=10,
                )
                stored_file = self.storage.save(
                    content,
                    filename=filename,
                    content_hash=content_hash,
                )
                document.storage_uri = stored_file.uri
                parsed = self.parser_registry.get(filename).parse(content)

                job_repository.set_status(
                    job,
                    IngestionJobStatus.CHUNKING,
                    progress=50,
                )
                chunks = self.chunker.chunk(parsed.text, source_id=filename)
                ChunkRepository(session).add_from_chunks(
                    document,
                    chunks,
                    metadata_for_chunk=lambda chunk: parsed.metadata_for_range(
                        chunk.metadata.char_start,
                        chunk.metadata.char_end,
                    ),
                )
                document.status = DocumentStatus.READY
                job.finished_at = datetime.now(UTC)
                job_repository.set_status(
                    job,
                    IngestionJobStatus.COMPLETED,
                    progress=100,
                )
            except Exception as error:
                message = str(error) or error.__class__.__name__
                document.status = DocumentStatus.FAILED
                document.error_message = message
                job.finished_at = datetime.now(UTC)
                job_repository.set_status(
                    job,
                    IngestionJobStatus.FAILED,
                    error_message=message,
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

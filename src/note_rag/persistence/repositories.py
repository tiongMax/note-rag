"""Repository layer for persistence operations."""

import uuid
from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from note_rag.chunking import Chunk
from note_rag.persistence.models import (
    ChunkRecord,
    Document,
    IngestionJob,
    IngestionJobStatus,
)


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, document: Document) -> Document:
        self.session.add(document)
        self.session.flush()
        return document

    def get(self, document_id: uuid.UUID) -> Document | None:
        return self.session.get(Document, document_id)

    def list(self, *, offset: int = 0, limit: int = 100) -> list[Document]:
        statement = (
            select(Document)
            .order_by(Document.created_at, Document.id)
            .offset(offset)
            .limit(limit)
        )
        return list(self.session.scalars(statement))

    def delete(self, document: Document) -> None:
        self.session.delete(document)
        self.session.flush()


class ChunkRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_from_chunks(
        self,
        document: Document,
        chunks: Iterable[Chunk],
    ) -> list[ChunkRecord]:
        records = [
            ChunkRecord(
                document=document,
                position=chunk.metadata.index,
                text=chunk.text,
                token_count=chunk.metadata.token_count,
                token_start=chunk.metadata.token_start,
                token_end=chunk.metadata.token_end,
                char_start=chunk.metadata.char_start,
                char_end=chunk.metadata.char_end,
                source_metadata=(
                    {"source_id": chunk.metadata.source_id}
                    if chunk.metadata.source_id is not None
                    else {}
                ),
            )
            for chunk in chunks
        ]
        self.session.add_all(records)
        document.chunk_count = len(records)
        document.token_count = max(
            (record.token_end for record in records),
            default=0,
        )
        self.session.flush()
        return records

    def list_for_document(self, document_id: uuid.UUID) -> list[ChunkRecord]:
        statement = (
            select(ChunkRecord)
            .where(ChunkRecord.document_id == document_id)
            .order_by(ChunkRecord.position)
        )
        return list(self.session.scalars(statement))


class IngestionJobRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, job: IngestionJob) -> IngestionJob:
        self.session.add(job)
        self.session.flush()
        return job

    def get(self, job_id: uuid.UUID) -> IngestionJob | None:
        return self.session.get(IngestionJob, job_id)

    def list_for_document(self, document_id: uuid.UUID) -> list[IngestionJob]:
        statement = (
            select(IngestionJob)
            .where(IngestionJob.document_id == document_id)
            .order_by(IngestionJob.created_at, IngestionJob.id)
        )
        return list(self.session.scalars(statement))

    def set_status(
        self,
        job: IngestionJob,
        status: IngestionJobStatus,
        *,
        progress: int | None = None,
        error_message: str | None = None,
    ) -> IngestionJob:
        if progress is not None and not 0 <= progress <= 100:
            raise ValueError("progress must be between 0 and 100")
        job.status = status
        if progress is not None:
            job.progress = progress
        job.error_message = error_message
        self.session.flush()
        return job

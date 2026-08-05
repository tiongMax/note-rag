"""Repository layer for persistence operations."""

import uuid
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import func, or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from note_rag.chunking import Chunk
from note_rag.persistence.models import (
    ChatMessageRecord,
    ChatRole,
    ChunkRecord,
    Conversation,
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

    def get_by_content_hash(self, content_hash: str) -> Document | None:
        statement = select(Document).where(Document.content_hash == content_hash)
        return self.session.scalar(statement)

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
        *,
        metadata_for_chunk: Callable[[Chunk], dict[str, Any]] | None = None,
    ) -> list[ChunkRecord]:
        records = []
        for chunk in chunks:
            source_metadata: dict[str, Any] = {}
            if chunk.metadata.source_id is not None:
                source_metadata["source_id"] = chunk.metadata.source_id
            if metadata_for_chunk is not None:
                source_metadata.update(metadata_for_chunk(chunk))
            records.append(
                ChunkRecord(
                    document=document,
                    position=chunk.metadata.index,
                    text=chunk.text,
                    token_count=chunk.metadata.token_count,
                    token_start=chunk.metadata.token_start,
                    token_end=chunk.metadata.token_end,
                    char_start=chunk.metadata.char_start,
                    char_end=chunk.metadata.char_end,
                    source_metadata=source_metadata,
                )
            )
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

    def claim_next(
        self,
        *,
        worker_id: str,
        now: datetime,
    ) -> IngestionJob | None:
        statement = (
            select(IngestionJob)
            .where(
                IngestionJob.status == IngestionJobStatus.QUEUED,
                or_(
                    IngestionJob.next_attempt_at.is_(None),
                    IngestionJob.next_attempt_at <= now,
                ),
            )
            .order_by(IngestionJob.created_at, IngestionJob.id)
            .limit(1)
        )
        if (
            self.session.bind is not None
            and self.session.bind.dialect.name == "postgresql"
        ):
            statement = statement.with_for_update(skip_locked=True)
        job = self.session.scalar(statement)
        if job is None:
            return None
        return self._claim(job, worker_id=worker_id, now=now)

    def claim(
        self,
        job_id: uuid.UUID,
        *,
        worker_id: str,
        now: datetime,
    ) -> IngestionJob | None:
        statement = select(IngestionJob).where(
            IngestionJob.id == job_id,
            IngestionJob.status == IngestionJobStatus.QUEUED,
            or_(
                IngestionJob.next_attempt_at.is_(None),
                IngestionJob.next_attempt_at <= now,
            ),
        )
        if (
            self.session.bind is not None
            and self.session.bind.dialect.name == "postgresql"
        ):
            statement = statement.with_for_update(skip_locked=True)
        job = self.session.scalar(statement)
        if job is None:
            return None
        return self._claim(job, worker_id=worker_id, now=now)

    def recover_stale(
        self,
        *,
        stale_before: datetime,
        now: datetime,
    ) -> int:
        active = (
            IngestionJobStatus.PARSING,
            IngestionJobStatus.CHUNKING,
            IngestionJobStatus.EMBEDDING,
            IngestionJobStatus.INDEXING,
        )
        result = self.session.execute(
            update(IngestionJob)
            .where(
                IngestionJob.status.in_(active),
                or_(
                    IngestionJob.locked_at.is_(None),
                    IngestionJob.locked_at < stale_before,
                ),
            )
            .values(
                status=IngestionJobStatus.QUEUED,
                next_attempt_at=now,
                locked_at=None,
                worker_id=None,
                error_message="recovered stale worker lease",
                finished_at=None,
            )
        )
        self.session.flush()
        return cast(CursorResult[Any], result).rowcount

    def reschedule(
        self,
        job: IngestionJob,
        *,
        error_message: str,
        next_attempt_at: datetime,
    ) -> None:
        job.status = IngestionJobStatus.QUEUED
        job.progress = 0
        job.error_message = error_message
        job.next_attempt_at = next_attempt_at
        job.locked_at = None
        job.worker_id = None
        job.finished_at = None
        self.session.flush()

    def fail(self, job: IngestionJob, *, error_message: str) -> None:
        job.status = IngestionJobStatus.FAILED
        job.error_message = error_message
        job.next_attempt_at = None
        job.locked_at = None
        job.worker_id = None
        job.finished_at = datetime.now(UTC)
        self.session.flush()

    def complete_claim(self, job: IngestionJob) -> None:
        job.status = IngestionJobStatus.COMPLETED
        job.progress = 100
        job.error_message = None
        job.next_attempt_at = None
        job.locked_at = None
        job.worker_id = None
        job.finished_at = datetime.now(UTC)
        self.session.flush()

    def _claim(
        self,
        job: IngestionJob,
        *,
        worker_id: str,
        now: datetime,
    ) -> IngestionJob:
        job.status = IngestionJobStatus.PARSING
        job.progress = 10
        job.attempts += 1
        job.error_message = None
        job.next_attempt_at = None
        job.locked_at = now
        job.worker_id = worker_id
        job.started_at = job.started_at or now
        self.session.flush()
        return job


class ConversationRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, conversation: Conversation) -> Conversation:
        self.session.add(conversation)
        self.session.flush()
        return conversation

    def get(self, conversation_id: uuid.UUID) -> Conversation | None:
        return self.session.get(Conversation, conversation_id)

    def list(self, *, offset: int = 0, limit: int = 100) -> list[Conversation]:
        statement = (
            select(Conversation)
            .order_by(Conversation.updated_at.desc(), Conversation.id)
            .offset(offset)
            .limit(limit)
        )
        return list(self.session.scalars(statement))

    def delete(self, conversation: Conversation) -> None:
        self.session.delete(conversation)
        self.session.flush()


class ChatMessageRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(
        self,
        conversation: Conversation,
        *,
        role: ChatRole,
        content: str,
        token_count: int,
        citations: list[dict[str, Any]] | None = None,
        context_token_count: int = 0,
        model_name: str | None = None,
    ) -> ChatMessageRecord:
        position = self.session.scalar(
            select(func.max(ChatMessageRecord.position)).where(
                ChatMessageRecord.conversation_id == conversation.id
            )
        )
        message = ChatMessageRecord(
            conversation=conversation,
            position=(position + 1 if position is not None else 0),
            role=role,
            content=content,
            token_count=token_count,
            citations=citations or [],
            context_token_count=context_token_count,
            model_name=model_name,
        )
        conversation.updated_at = datetime.now(UTC)
        self.session.add(message)
        self.session.flush()
        return message

    def list_for_conversation(
        self,
        conversation_id: uuid.UUID,
    ) -> list[ChatMessageRecord]:
        statement = (
            select(ChatMessageRecord)
            .where(ChatMessageRecord.conversation_id == conversation_id)
            .order_by(ChatMessageRecord.position)
        )
        return list(self.session.scalars(statement))

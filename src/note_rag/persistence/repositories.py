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
    ApprovalStatus,
    ChatMessageRecord,
    ChatRole,
    ChunkRecord,
    Conversation,
    Course,
    CourseDocument,
    Document,
    GenerationJob,
    GenerationJobStatus,
    IngestionJob,
    IngestionJobStatus,
    LearningObjective,
    MasteryScope,
    MasterySnapshot,
    MemoryState,
    ReviewAttempt,
    StudyItem,
    StudyItemSource,
    StudySession,
    StudySessionItem,
    StudySessionStatus,
    Topic,
    TopicSource,
    TopicState,
)


class GenerationJobRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, job: GenerationJob) -> GenerationJob:
        existing = self.session.scalar(
            select(GenerationJob).where(
                GenerationJob.idempotency_key == job.idempotency_key
            )
        )
        if existing is not None:
            return existing
        self.session.add(job)
        self.session.flush()
        return job

    def get(self, job_id: uuid.UUID) -> GenerationJob | None:
        return self.session.get(GenerationJob, job_id)

    def claim_next(self) -> GenerationJob | None:
        statement = (
            select(GenerationJob)
            .where(GenerationJob.status == GenerationJobStatus.QUEUED)
            .order_by(GenerationJob.created_at, GenerationJob.id)
        )
        if (
            self.session.bind is not None
            and self.session.bind.dialect.name == "postgresql"
        ):
            statement = statement.with_for_update(skip_locked=True)
        job = self.session.scalar(statement.limit(1))
        if job is not None:
            job.status = GenerationJobStatus.RUNNING
            job.progress = 10
            job.attempts += 1
            job.started_at = datetime.now(UTC)
            self.session.flush()
        return job

    def claim(self, job_id: uuid.UUID) -> GenerationJob | None:
        job = self.get(job_id)
        if job is None or job.status is not GenerationJobStatus.QUEUED:
            return None
        job.status = GenerationJobStatus.RUNNING
        job.progress = 10
        job.attempts += 1
        job.started_at = datetime.now(UTC)
        self.session.flush()
        return job

    def complete(self, job: GenerationJob) -> None:
        job.status = GenerationJobStatus.COMPLETED
        job.progress = 100
        job.error_message = None
        job.finished_at = datetime.now(UTC)
        self.session.flush()

    def fail(self, job: GenerationJob, message: str, *, retry: bool) -> None:
        job.status = GenerationJobStatus.QUEUED if retry else GenerationJobStatus.FAILED
        job.progress = 0 if retry else job.progress
        job.error_message = message
        if not retry:
            job.finished_at = datetime.now(UTC)
        self.session.flush()


class StudyItemRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, item_id: uuid.UUID) -> StudyItem | None:
        return self.session.get(StudyItem, item_id)

    def list_for_topic(
        self, topic_id: uuid.UUID, *, include_archived: bool = False
    ) -> list[StudyItem]:
        statement = select(StudyItem).where(StudyItem.topic_id == topic_id)
        if not include_archived:
            statement = statement.where(
                StudyItem.approval_status != ApprovalStatus.ARCHIVED
            )
        return list(
            self.session.scalars(statement.order_by(StudyItem.created_at, StudyItem.id))
        )

    def add(self, item: StudyItem, chunk_ids: list[uuid.UUID]) -> StudyItem:
        self.session.add(item)
        self.session.flush()
        self.session.add_all(
            [
                StudyItemSource(study_item=item, chunk_id=chunk_id)
                for chunk_id in dict.fromkeys(chunk_ids)
            ]
        )
        self.session.flush()
        return item

    def add_objective(self, objective: LearningObjective) -> LearningObjective:
        self.session.add(objective)
        self.session.flush()
        return objective

    def list_approved_for_course(self, course_id: uuid.UUID) -> list[StudyItem]:
        return list(
            self.session.scalars(
                select(StudyItem)
                .join(Topic)
                .where(
                    Topic.course_id == course_id,
                    StudyItem.approval_status == ApprovalStatus.APPROVED,
                )
                .order_by(Topic.position, StudyItem.created_at, StudyItem.id)
            )
        )


class StudySessionRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(
        self, study_session: StudySession, items: list[tuple[StudyItem, str]]
    ) -> StudySession:
        self.session.add(study_session)
        self.session.flush()
        for position, (item, reason) in enumerate(items):
            self.session.add(
                StudySessionItem(
                    session_id=study_session.id,
                    study_item_id=item.id,
                    position=position,
                    reason=reason,
                )
            )
        self.session.flush()
        return study_session

    def get(self, session_id: uuid.UUID) -> StudySession | None:
        return self.session.get(StudySession, session_id)

    def complete(self, study_session: StudySession, at: datetime) -> None:
        study_session.status = StudySessionStatus.COMPLETED
        study_session.completed_at = at
        self.session.flush()


class ReviewAttemptRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, attempt_id: uuid.UUID) -> ReviewAttempt | None:
        return self.session.get(ReviewAttempt, attempt_id)

    def get_by_key(
        self, session_id: uuid.UUID, idempotency_key: str
    ) -> ReviewAttempt | None:
        return self.session.scalar(
            select(ReviewAttempt).where(
                ReviewAttempt.session_id == session_id,
                ReviewAttempt.idempotency_key == idempotency_key,
            )
        )

    def add(self, attempt: ReviewAttempt) -> ReviewAttempt:
        self.session.add(attempt)
        self.session.flush()
        return attempt

    def list_for_item(self, item_id: uuid.UUID) -> list[ReviewAttempt]:
        return list(
            self.session.scalars(
                select(ReviewAttempt)
                .where(ReviewAttempt.study_item_id == item_id)
                .order_by(ReviewAttempt.reviewed_at, ReviewAttempt.id)
            )
        )


class MemoryStateRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, item_id: uuid.UUID) -> MemoryState | None:
        return self.session.get(MemoryState, item_id)

    def save(self, state: MemoryState) -> MemoryState:
        self.session.add(state)
        self.session.flush()
        return state


class MasterySnapshotRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, snapshot: MasterySnapshot) -> MasterySnapshot:
        self.session.add(snapshot)
        self.session.flush()
        return snapshot

    def list_for_course(
        self, course_id: uuid.UUID, *, limit: int = 365
    ) -> list[MasterySnapshot]:
        return list(
            self.session.scalars(
                select(MasterySnapshot)
                .where(
                    MasterySnapshot.course_id == course_id,
                    MasterySnapshot.scope == MasteryScope.COURSE,
                )
                .order_by(MasterySnapshot.captured_at.desc(), MasterySnapshot.id.desc())
                .limit(limit)
            )
        )[::-1]

    def list_for_topic(
        self, topic_id: uuid.UUID, *, limit: int = 365
    ) -> list[MasterySnapshot]:
        return list(
            self.session.scalars(
                select(MasterySnapshot)
                .where(
                    MasterySnapshot.topic_id == topic_id,
                    MasterySnapshot.scope == MasteryScope.TOPIC,
                )
                .order_by(MasterySnapshot.captured_at.desc(), MasterySnapshot.id.desc())
                .limit(limit)
            )
        )[::-1]


class CourseRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, course: Course) -> Course:
        self.session.add(course)
        self.session.flush()
        return course

    def get(self, course_id: uuid.UUID) -> Course | None:
        return self.session.get(Course, course_id)

    def list(self, *, offset: int = 0, limit: int = 100) -> list[Course]:
        statement = (
            select(Course)
            .order_by(Course.created_at, Course.id)
            .offset(offset)
            .limit(limit)
        )
        return list(self.session.scalars(statement))

    def attach_document(self, course: Course, document: Document) -> CourseDocument:
        existing = self.session.get(CourseDocument, (course.id, document.id))
        if existing is not None:
            return existing
        link = CourseDocument(course=course, document=document)
        self.session.add(link)
        self.session.flush()
        return link

    def detach_document(self, course_id: uuid.UUID, document_id: uuid.UUID) -> bool:
        link = self.session.get(CourseDocument, (course_id, document_id))
        if link is None:
            return False
        self.session.delete(link)
        self.session.flush()
        return True

    def delete(self, course: Course) -> None:
        self.session.delete(course)
        self.session.flush()


class TopicRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, topic_id: uuid.UUID) -> Topic | None:
        return self.session.get(Topic, topic_id)

    def list_for_course(self, course_id: uuid.UUID) -> list[Topic]:
        statement = (
            select(Topic)
            .where(Topic.course_id == course_id)
            .order_by(Topic.parent_id, Topic.position, Topic.id)
        )
        return list(self.session.scalars(statement))

    def add(
        self,
        course: Course,
        *,
        title: str,
        description: str = "",
        parent: Topic | None = None,
        state: TopicState = TopicState.DRAFT,
        position: int | None = None,
    ) -> Topic:
        self._validate_parent(course.id, parent)
        siblings = self._siblings(course.id, parent.id if parent else None)
        resolved_position = len(siblings) if position is None else position
        if not 0 <= resolved_position <= len(siblings):
            raise ValueError("position is outside the sibling range")
        self._shift(siblings[resolved_position:], 1)
        topic = Topic(
            course=course,
            parent=parent,
            title=title,
            description=description,
            state=state,
            position=resolved_position,
        )
        self.session.add(topic)
        self.session.flush()
        return topic

    def update(
        self,
        topic: Topic,
        *,
        title: str | None = None,
        description: str | None = None,
        state: TopicState | None = None,
        parent: Topic | None | object = ...,
        position: int | None = None,
    ) -> Topic:
        old_parent_id = topic.parent_id
        new_parent = topic.parent if parent is ... else cast(Topic | None, parent)
        self._validate_parent(topic.course_id, new_parent, topic=topic)
        new_parent_id = new_parent.id if new_parent else None
        if title is not None:
            topic.title = title
        if description is not None:
            topic.description = description
        if state is not None:
            topic.state = state
        if new_parent_id != old_parent_id or position is not None:
            old_siblings = self._siblings(
                topic.course_id, old_parent_id, exclude=topic.id
            )
            self._renumber(old_siblings)
            new_siblings = self._siblings(
                topic.course_id, new_parent_id, exclude=topic.id
            )
            resolved = len(new_siblings) if position is None else position
            if not 0 <= resolved <= len(new_siblings):
                raise ValueError("position is outside the sibling range")
            topic.parent = new_parent
            new_siblings.insert(resolved, topic)
            self._renumber(new_siblings)
        self.session.flush()
        return topic

    def add_source(self, topic: Topic, chunk: ChunkRecord) -> TopicSource:
        if chunk.document_id not in {
            link.document_id for link in topic.course.document_links
        }:
            raise ValueError("source chunk document is not attached to the course")
        source = self.session.get(TopicSource, (topic.id, chunk.id))
        if source is None:
            source = TopicSource(topic=topic, chunk=chunk)
            self.session.add(source)
            self.session.flush()
        return source

    def delete(self, topic: Topic) -> None:
        siblings = self._siblings(topic.course_id, topic.parent_id, exclude=topic.id)
        self.session.delete(topic)
        self.session.flush()
        self._renumber(siblings)
        self.session.flush()

    def _validate_parent(
        self, course_id: uuid.UUID, parent: Topic | None, *, topic: Topic | None = None
    ) -> None:
        if parent is None:
            return
        if parent.course_id != course_id:
            raise ValueError("parent topic must belong to the same course")
        current: Topic | None = parent
        while current is not None:
            if topic is not None and current.id == topic.id:
                raise ValueError("topic hierarchy cannot contain a cycle")
            current = current.parent

    def _siblings(
        self,
        course_id: uuid.UUID,
        parent_id: uuid.UUID | None,
        *,
        exclude: uuid.UUID | None = None,
    ) -> list[Topic]:
        statement = select(Topic).where(
            Topic.course_id == course_id, Topic.parent_id == parent_id
        )
        if exclude is not None:
            statement = statement.where(Topic.id != exclude)
        return list(self.session.scalars(statement.order_by(Topic.position, Topic.id)))

    @staticmethod
    def _shift(topics: list[Topic], amount: int) -> None:
        for topic in reversed(topics):
            topic.position += amount

    @staticmethod
    def _renumber(topics: list[Topic]) -> None:
        for position, topic in enumerate(topics):
            topic.position = position


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

    def get(self, chunk_id: uuid.UUID) -> ChunkRecord | None:
        return self.session.get(ChunkRecord, chunk_id)

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

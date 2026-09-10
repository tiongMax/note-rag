"""Relational models for documents, chunks, and ingestion jobs."""

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any, cast

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
    literal_column,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class DocumentStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class IngestionJobStatus(StrEnum):
    QUEUED = "queued"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


class IndexingStatus(StrEnum):
    PENDING = "pending"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


class ChatRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class TopicState(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"


class GenerationJobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerationKind(StrEnum):
    CURRICULUM = "curriculum"
    STUDY_ITEMS = "study_items"
    STUDY_ITEM = "study_item"


class StudyItemType(StrEnum):
    FLASHCARD = "flashcard"
    MULTIPLE_CHOICE = "multiple_choice"
    SHORT_ANSWER = "short_answer"


class ApprovalStatus(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"
    ARCHIVED = "archived"


class StudySessionStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"


class StudySessionMode(StrEnum):
    DAILY_REVIEW = "daily_review"
    COURSE = "course"
    TOPIC = "topic"
    SELECTED = "selected"


class ReviewRating(StrEnum):
    AGAIN = "again"
    HARD = "hard"
    GOOD = "good"
    EASY = "easy"


class Document(TimestampMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint("token_count >= 0", name="ck_documents_token_count"),
        CheckConstraint("chunk_count >= 0", name="ck_documents_chunk_count"),
        Index(
            "uq_documents_content_hash",
            "content_hash",
            unique=True,
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    media_type: Mapped[str] = mapped_column(String(128), nullable=False)
    storage_uri: Mapped[str | None] = mapped_column(String(1024))
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, name="document_status"),
        default=DocumentStatus.PENDING,
        nullable=False,
        index=True,
    )
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    indexing_status: Mapped[IndexingStatus] = mapped_column(
        Enum(IndexingStatus, name="indexing_status"),
        default=IndexingStatus.PENDING,
        nullable=False,
        index=True,
    )
    embedding_model: Mapped[str | None] = mapped_column(String(255))
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    indexing_error: Mapped[str | None] = mapped_column(Text)

    chunks: Mapped[list["ChunkRecord"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="ChunkRecord.position",
    )
    ingestion_jobs: Mapped[list["IngestionJob"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="IngestionJob.created_at",
    )
    course_links: Mapped[list["CourseDocument"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


class ChunkRecord(TimestampMixin, Base):
    __tablename__ = "chunks"
    __table_args__ = (
        CheckConstraint("position >= 0", name="ck_chunks_position"),
        CheckConstraint("token_count > 0", name="ck_chunks_token_count"),
        CheckConstraint(
            "token_start >= 0 AND token_end > token_start",
            name="ck_chunks_token_range",
        ),
        CheckConstraint(
            "char_start >= 0 AND char_end > char_start",
            name="ck_chunks_char_range",
        ),
        UniqueConstraint(
            "document_id",
            "position",
            name="uq_chunks_document_position",
        ),
        Index("ix_chunks_document_token_start", "document_id", "token_start"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index(
            "ix_chunks_source_metadata_gin",
            "source_metadata",
            postgresql_using="gin",
        ).ddl_if(dialect="postgresql"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    token_start: Mapped[int] = mapped_column(Integer, nullable=False)
    token_end: Mapped[int] = mapped_column(Integer, nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    source_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        default=dict,
        nullable=False,
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(768))
    embedding_model: Mapped[str | None] = mapped_column(String(255))
    embedded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    document: Mapped[Document] = relationship(back_populates="chunks")
    topic_links: Mapped[list["TopicSource"]] = relationship(
        back_populates="chunk",
        cascade="all, delete-orphan",
    )


chunk_text_fts_index = Index(
    "ix_chunks_text_fts",
    func.to_tsvector(
        literal_column("'english'::regconfig"),
        ChunkRecord.text,
    ),
    postgresql_using="gin",
)
cast(Table, ChunkRecord.__table__).append_constraint(
    chunk_text_fts_index.ddl_if(dialect="postgresql")
)


class Course(TimestampMixin, Base):
    __tablename__ = "courses"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)

    document_links: Mapped[list["CourseDocument"]] = relationship(
        back_populates="course",
        cascade="all, delete-orphan",
        order_by="CourseDocument.created_at",
    )
    topics: Mapped[list["Topic"]] = relationship(
        back_populates="course",
        cascade="all, delete-orphan",
        order_by="Topic.position",
    )
    study_sessions: Mapped[list["StudySession"]] = relationship(
        back_populates="course", cascade="all, delete-orphan"
    )


class CourseDocument(TimestampMixin, Base):
    __tablename__ = "course_documents"

    course_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("courses.id", ondelete="CASCADE"), primary_key=True
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )

    course: Mapped[Course] = relationship(back_populates="document_links")
    document: Mapped[Document] = relationship(back_populates="course_links")


class Topic(TimestampMixin, Base):
    __tablename__ = "topics"
    __table_args__ = (
        CheckConstraint("position >= 0", name="ck_topics_position"),
        UniqueConstraint(
            "course_id", "parent_id", "position", name="uq_topics_sibling_position"
        ),
        Index("ix_topics_course_parent_position", "course_id", "parent_id", "position"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    course_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("courses.id", ondelete="CASCADE"), nullable=False
    )
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("topics.id", ondelete="CASCADE")
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    state: Mapped[TopicState] = mapped_column(
        Enum(TopicState, name="topic_state"),
        default=TopicState.DRAFT,
        nullable=False,
    )

    course: Mapped[Course] = relationship(back_populates="topics")
    parent: Mapped["Topic | None"] = relationship(
        back_populates="children", remote_side="Topic.id"
    )
    children: Mapped[list["Topic"]] = relationship(
        back_populates="parent",
        cascade="all, delete-orphan",
        order_by="Topic.position",
        single_parent=True,
    )
    sources: Mapped[list["TopicSource"]] = relationship(
        back_populates="topic", cascade="all, delete-orphan"
    )
    objectives: Mapped[list["LearningObjective"]] = relationship(
        back_populates="topic",
        cascade="all, delete-orphan",
        order_by="LearningObjective.position",
    )
    study_items: Mapped[list["StudyItem"]] = relationship(
        back_populates="topic", cascade="all, delete-orphan"
    )


class TopicSource(TimestampMixin, Base):
    __tablename__ = "topic_sources"

    topic_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("topics.id", ondelete="CASCADE"), primary_key=True
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )

    topic: Mapped[Topic] = relationship(back_populates="sources")
    chunk: Mapped[ChunkRecord] = relationship(back_populates="topic_links")


class LearningObjective(TimestampMixin, Base):
    __tablename__ = "learning_objectives"
    __table_args__ = (
        CheckConstraint("position >= 0", name="ck_learning_objectives_position"),
        UniqueConstraint(
            "topic_id", "position", name="uq_learning_objectives_topic_position"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    topic_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("topics.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    generation_version: Mapped[str | None] = mapped_column(String(128))

    topic: Mapped[Topic] = relationship(back_populates="objectives")
    study_items: Mapped[list["StudyItem"]] = relationship(back_populates="objective")


class StudyItem(TimestampMixin, Base):
    __tablename__ = "study_items"
    __table_args__ = (
        CheckConstraint(
            "difficulty >= 1 AND difficulty <= 5", name="ck_study_items_difficulty"
        ),
        Index("ix_study_items_topic_status", "topic_id", "approval_status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    topic_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("topics.id", ondelete="CASCADE"), nullable=False
    )
    objective_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("learning_objectives.id", ondelete="SET NULL")
    )
    item_type: Mapped[StudyItemType] = mapped_column(
        Enum(StudyItemType, name="study_item_type"), nullable=False
    )
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    options: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, default=list, nullable=False
    )
    difficulty: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    approval_status: Mapped[ApprovalStatus] = mapped_column(
        Enum(ApprovalStatus, name="approval_status"),
        default=ApprovalStatus.DRAFT,
        nullable=False,
    )
    generation_version: Mapped[str | None] = mapped_column(String(128))
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    topic: Mapped[Topic] = relationship(back_populates="study_items")
    objective: Mapped[LearningObjective | None] = relationship(
        back_populates="study_items"
    )
    sources: Mapped[list["StudyItemSource"]] = relationship(
        back_populates="study_item", cascade="all, delete-orphan"
    )
    review_attempts: Mapped[list["ReviewAttempt"]] = relationship(
        back_populates="study_item", cascade="all, delete-orphan"
    )
    memory_state: Mapped["MemoryState | None"] = relationship(
        back_populates="study_item", cascade="all, delete-orphan", uselist=False
    )


class StudyItemSource(TimestampMixin, Base):
    __tablename__ = "study_item_sources"

    study_item_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("study_items.id", ondelete="CASCADE"), primary_key=True
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    study_item: Mapped[StudyItem] = relationship(back_populates="sources")
    chunk: Mapped[ChunkRecord] = relationship()


class StudySession(TimestampMixin, Base):
    __tablename__ = "study_sessions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    course_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("courses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    topic_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("topics.id", ondelete="SET NULL")
    )
    mode: Mapped[StudySessionMode] = mapped_column(
        Enum(StudySessionMode, name="study_session_mode"), nullable=False
    )
    status: Mapped[StudySessionStatus] = mapped_column(
        Enum(StudySessionStatus, name="study_session_status"),
        default=StudySessionStatus.ACTIVE,
        nullable=False,
        index=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    course: Mapped[Course] = relationship(back_populates="study_sessions")
    items: Mapped[list["StudySessionItem"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="StudySessionItem.position",
    )
    attempts: Mapped[list["ReviewAttempt"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class StudySessionItem(Base):
    __tablename__ = "study_session_items"
    __table_args__ = (
        CheckConstraint("position >= 0", name="ck_study_session_items_position"),
        UniqueConstraint("session_id", "position", name="uq_session_item_position"),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("study_sessions.id", ondelete="CASCADE"), primary_key=True
    )
    study_item_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("study_items.id", ondelete="CASCADE"), primary_key=True
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    reason: Mapped[str] = mapped_column(String(255), nullable=False)

    session: Mapped[StudySession] = relationship(back_populates="items")
    study_item: Mapped[StudyItem] = relationship()


class ReviewAttempt(Base):
    __tablename__ = "review_attempts"
    __table_args__ = (
        CheckConstraint("score >= 0 AND score <= 1", name="ck_attempt_score"),
        CheckConstraint(
            "confidence >= 1 AND confidence <= 5", name="ck_attempt_confidence"
        ),
        CheckConstraint("response_time_ms >= 0", name="ck_attempt_response_time"),
        UniqueConstraint(
            "session_id", "idempotency_key", name="uq_attempt_session_idempotency"
        ),
        Index("ix_review_attempts_item_reviewed", "study_item_id", "reviewed_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("study_sessions.id", ondelete="CASCADE"), nullable=False
    )
    study_item_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("study_items.id", ondelete="CASCADE"), nullable=False
    )
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    submitted_answer: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str] = mapped_column(Text, nullable=False)
    correct: Mapped[bool] = mapped_column(nullable=False)
    score: Mapped[float] = mapped_column(nullable=False)
    rating: Mapped[ReviewRating] = mapped_column(
        Enum(ReviewRating, name="review_rating"), nullable=False
    )
    confidence: Mapped[int] = mapped_column(Integer, nullable=False)
    response_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    hint_used: Mapped[bool] = mapped_column(default=False, nullable=False)
    grading_details: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )
    reviewed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    overridden_correct: Mapped[bool | None] = mapped_column()
    overridden_score: Mapped[float | None] = mapped_column()
    override_reason: Mapped[str | None] = mapped_column(Text)
    overridden_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    session: Mapped[StudySession] = relationship(back_populates="attempts")
    study_item: Mapped[StudyItem] = relationship(back_populates="review_attempts")


class MemoryState(TimestampMixin, Base):
    __tablename__ = "memory_states"
    __table_args__ = (
        CheckConstraint("half_life_days > 0", name="ck_memory_half_life"),
        CheckConstraint(
            "difficulty >= 1 AND difficulty <= 5", name="ck_memory_difficulty"
        ),
        CheckConstraint(
            "predicted_recall >= 0 AND predicted_recall <= 1",
            name="ck_memory_predicted_recall",
        ),
        CheckConstraint(
            "successful_reviews >= 0 AND failed_reviews >= 0",
            name="ck_memory_review_counts",
        ),
        Index("ix_memory_states_next_review", "next_review_at"),
    )

    study_item_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("study_items.id", ondelete="CASCADE"), primary_key=True
    )
    half_life_days: Mapped[float] = mapped_column(nullable=False)
    difficulty: Mapped[int] = mapped_column(Integer, nullable=False)
    last_review_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    next_review_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    predicted_recall: Mapped[float] = mapped_column(nullable=False)
    successful_reviews: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_reviews: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    scheduler_version: Mapped[str] = mapped_column(String(64), nullable=False)

    study_item: Mapped[StudyItem] = relationship(back_populates="memory_state")


class GenerationJob(TimestampMixin, Base):
    __tablename__ = "generation_jobs"
    __table_args__ = (
        CheckConstraint(
            "progress >= 0 AND progress <= 100", name="ck_generation_jobs_progress"
        ),
        CheckConstraint("attempts >= 0", name="ck_generation_jobs_attempts"),
        UniqueConstraint("idempotency_key", name="uq_generation_jobs_idempotency_key"),
        Index("ix_generation_jobs_status_created", "status", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    course_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("courses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    topic_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("topics.id", ondelete="CASCADE")
    )
    item_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("study_items.id", ondelete="SET NULL")
    )
    kind: Mapped[GenerationKind] = mapped_column(
        Enum(GenerationKind, name="generation_kind"), nullable=False
    )
    status: Mapped[GenerationJobStatus] = mapped_column(
        Enum(GenerationJobStatus, name="generation_job_status"),
        default=GenerationJobStatus.QUEUED,
        nullable=False,
        index=True,
    )
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(128), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class IngestionJob(TimestampMixin, Base):
    __tablename__ = "ingestion_jobs"
    __table_args__ = (
        CheckConstraint(
            "progress >= 0 AND progress <= 100",
            name="ck_ingestion_jobs_progress",
        ),
        CheckConstraint("attempts >= 0", name="ck_ingestion_jobs_attempts"),
        Index("ix_ingestion_jobs_status_created", "status", "created_at"),
        Index(
            "ix_ingestion_jobs_status_next_attempt",
            "status",
            "next_attempt_at",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[IngestionJobStatus] = mapped_column(
        Enum(IngestionJobStatus, name="ingestion_job_status"),
        default=IngestionJobStatus.QUEUED,
        nullable=False,
    )
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    worker_id: Mapped[str | None] = mapped_column(String(64))

    document: Mapped[Document] = relationship(back_populates="ingestion_jobs")


class Conversation(TimestampMixin, Base):
    __tablename__ = "conversations"
    __table_args__ = (Index("ix_conversations_updated_at", "updated_at"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(255), nullable=False)

    messages: Mapped[list["ChatMessageRecord"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ChatMessageRecord.position",
    )


class ChatMessageRecord(TimestampMixin, Base):
    __tablename__ = "chat_messages"
    __table_args__ = (
        CheckConstraint(
            "token_count >= 0",
            name="ck_chat_messages_token_count",
        ),
        CheckConstraint(
            "context_token_count >= 0",
            name="ck_chat_messages_context_token_count",
        ),
        CheckConstraint(
            "position >= 0",
            name="ck_chat_messages_position",
        ),
        UniqueConstraint(
            "conversation_id",
            "position",
            name="uq_chat_messages_conversation_position",
        ),
        Index(
            "ix_chat_messages_conversation_created",
            "conversation_id",
            "created_at",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[ChatRole] = mapped_column(
        Enum(ChatRole, name="chat_role"),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        default=list,
        nullable=False,
    )
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    context_token_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    model_name: Mapped[str | None] = mapped_column(String(255))

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class CacheState(TimestampMixin, Base):
    """Small persistent state used to version corpus-dependent cache entries."""

    __tablename__ = "cache_state"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[int] = mapped_column(BigInteger, nullable=False)


class QueryEmbeddingCache(TimestampMixin, Base):
    __tablename__ = "query_embedding_cache"
    __table_args__ = (Index("ix_query_embedding_cache_expires", "expires_at"),)

    key_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_query: Mapped[str] = mapped_column(Text, nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    vector: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )


class RetrievalResultCache(TimestampMixin, Base):
    __tablename__ = "retrieval_result_cache"
    __table_args__ = (
        Index("ix_retrieval_result_cache_expires", "expires_at"),
        Index("ix_retrieval_result_cache_corpus", "corpus_version"),
    )

    key_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    corpus_version: Mapped[int] = mapped_column(BigInteger, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

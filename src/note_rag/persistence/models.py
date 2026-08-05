"""Relational models for documents, chunks, and ingestion jobs."""

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any, cast

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
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


class IngestionJob(TimestampMixin, Base):
    __tablename__ = "ingestion_jobs"
    __table_args__ = (
        CheckConstraint(
            "progress >= 0 AND progress <= 100",
            name="ck_ingestion_jobs_progress",
        ),
        CheckConstraint("attempts >= 0", name="ck_ingestion_jobs_attempts"),
        Index("ix_ingestion_jobs_status_created", "status", "created_at"),
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

    document: Mapped[Document] = relationship(back_populates="ingestion_jobs")


class Conversation(TimestampMixin, Base):
    __tablename__ = "conversations"
    __table_args__ = (
        Index("ix_conversations_updated_at", "updated_at"),
    )

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

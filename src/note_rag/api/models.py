"""Phase 1 HTTP request and response contracts."""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from note_rag.chunking.models import Chunk
from note_rag.persistence import DocumentStatus, IngestionJobStatus


class ChunkTextRequest(BaseModel):
    text: str
    source_id: str | None = None
    chunk_size: int | None = Field(default=None, gt=0)
    chunk_overlap: int | None = Field(default=None, ge=0)


class ChunkTextResponse(BaseModel):
    token_count: int
    chunks: list[Chunk]


class IngestionResponse(BaseModel):
    document_id: uuid.UUID
    job_id: uuid.UUID | None
    status: DocumentStatus
    duplicate: bool
    chunk_count: int
    token_count: int
    error_message: str | None = None


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    media_type: str
    storage_uri: str | None
    content_hash: str | None
    status: DocumentStatus
    token_count: int
    chunk_count: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class StoredChunkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_id: uuid.UUID
    position: int
    text: str
    token_count: int
    token_start: int
    token_end: int
    char_start: int
    char_end: int
    source_metadata: dict[str, Any]


class IngestionJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_id: uuid.UUID
    status: IngestionJobStatus
    progress: int
    attempts: int
    error_message: str | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime

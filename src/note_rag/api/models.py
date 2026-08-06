"""HTTP request and response contracts."""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from note_rag.chunking.models import Chunk
from note_rag.persistence import (
    ChatRole,
    DocumentStatus,
    IndexingStatus,
    IngestionJobStatus,
)
from note_rag.retrieval import SearchMode


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
    indexing_status: IndexingStatus
    indexing_error: str | None = None


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
    indexing_status: IndexingStatus
    embedding_model: str | None
    indexed_at: datetime | None
    indexing_error: str | None
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
    next_attempt_at: datetime | None
    locked_at: datetime | None
    worker_id: str | None
    created_at: datetime
    updated_at: datetime


class IndexingResponse(BaseModel):
    document_id: uuid.UUID
    status: IndexingStatus
    indexed_chunks: int
    embedding_model: str
    error_message: str | None = None


class SearchFiltersRequest(BaseModel):
    document_ids: list[uuid.UUID] = Field(default_factory=list)
    filenames: list[str] = Field(default_factory=list)
    media_types: list[str] = Field(default_factory=list)
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: SearchMode = SearchMode.HYBRID
    top_k: int = Field(default=10, ge=1, le=100)
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    filters: SearchFiltersRequest = Field(default_factory=SearchFiltersRequest)


class SearchHitResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    media_type: str
    position: int
    text: str
    token_count: int
    token_start: int
    token_end: int
    char_start: int
    char_end: int
    source_metadata: dict[str, Any]
    score: float
    vector_score: float | None
    keyword_score: float | None


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    hits: list[SearchHitResponse]


class ContextRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: SearchMode = SearchMode.HYBRID
    candidate_k: int | None = Field(default=None, ge=1, le=100)
    max_chunks: int | None = Field(default=None, ge=1, le=100)
    max_context_tokens: int | None = Field(default=None, ge=1)
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    rerank: bool = True
    rerank_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    filters: SearchFiltersRequest = Field(default_factory=SearchFiltersRequest)


class ContextChunkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    citation_id: int
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    media_type: str
    position: int
    text: str
    token_count: int
    source_metadata: dict[str, Any]
    retrieval_score: float
    rerank_score: float | None
    score: float
    truncated: bool


class ContextResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    query: str
    mode: SearchMode
    context: str
    chunks: list[ContextChunkResponse]
    token_count: int
    token_budget: int
    candidates_considered: int
    duplicates_removed: int
    truncated: bool
    reranker_model: str | None


class ChatRequest(ContextRequest):
    conversation_id: uuid.UUID | None = None


class CitationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    citation_id: int
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    position: int
    source_metadata: dict[str, Any]


class ChatResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    conversation_id: uuid.UUID
    message_id: uuid.UUID
    answer: str
    citations: list[CitationResponse]
    model_name: str


class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    conversation_id: uuid.UUID
    position: int
    role: ChatRole
    content: str
    citations: list[CitationResponse]
    token_count: int
    context_token_count: int
    model_name: str | None
    created_at: datetime


class ConversationResponse(BaseModel):
    id: uuid.UUID
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime


class ConversationDetailResponse(ConversationResponse):
    messages: list[ChatMessageResponse]

"""HTTP request and response contracts."""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from note_rag.chunking.models import Chunk
from note_rag.persistence import (
    ApprovalStatus,
    ChatRole,
    DocumentStatus,
    GenerationJobStatus,
    GenerationKind,
    IndexingStatus,
    IngestionJobStatus,
    ReviewRating,
    StudyItemType,
    StudySessionMode,
    StudySessionStatus,
    TopicState,
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


class CourseCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    description: str = Field(default="", max_length=5000)


class CourseUpdateRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=5000)


class CourseDocumentsRequest(BaseModel):
    document_ids: list[uuid.UUID] = Field(min_length=1)


class CourseResponse(BaseModel):
    id: uuid.UUID
    title: str
    description: str
    document_ids: list[uuid.UUID]
    topic_count: int
    created_at: datetime
    updated_at: datetime


class TopicCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    description: str = Field(default="", max_length=5000)
    parent_id: uuid.UUID | None = None
    position: int | None = Field(default=None, ge=0)
    state: TopicState = TopicState.DRAFT
    source_chunk_ids: list[uuid.UUID] = Field(default_factory=list)


class TopicUpdateRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=5000)
    parent_id: uuid.UUID | None = None
    position: int | None = Field(default=None, ge=0)
    state: TopicState | None = None


class TopicMergeRequest(BaseModel):
    target_topic_id: uuid.UUID


class TopicResponse(BaseModel):
    id: uuid.UUID
    course_id: uuid.UUID
    parent_id: uuid.UUID | None
    title: str
    description: str
    position: int
    state: TopicState
    source_chunk_ids: list[uuid.UUID]
    created_at: datetime
    updated_at: datetime


class GenerationRequest(BaseModel):
    idempotency_key: str | None = Field(default=None, min_length=8, max_length=128)


class GenerationJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    course_id: uuid.UUID
    topic_id: uuid.UUID | None
    item_id: uuid.UUID | None
    kind: GenerationKind
    status: GenerationJobStatus
    progress: int
    attempts: int
    prompt_version: str
    model_name: str
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class StudyOptionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    correct: bool


class StudyItemUpdateRequest(BaseModel):
    prompt: str | None = Field(default=None, min_length=1, max_length=4000)
    answer: str | None = Field(default=None, min_length=1, max_length=4000)
    explanation: str | None = Field(default=None, min_length=1, max_length=6000)
    difficulty: int | None = Field(default=None, ge=1, le=5)
    options: list[StudyOptionRequest] | None = Field(default=None, max_length=8)
    approval_status: ApprovalStatus | None = None


class SourcePassageResponse(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    position: int
    text: str


class StudyItemResponse(BaseModel):
    id: uuid.UUID
    topic_id: uuid.UUID
    objective_id: uuid.UUID | None
    item_type: StudyItemType
    prompt: str
    answer: str
    explanation: str
    options: list[dict[str, Any]]
    difficulty: int
    approval_status: ApprovalStatus
    generation_version: str | None
    sources: list[SourcePassageResponse]
    created_at: datetime
    updated_at: datetime


class StudySessionCreateRequest(BaseModel):
    course_id: uuid.UUID
    topic_id: uuid.UUID | None = None
    item_ids: list[uuid.UUID] = Field(default_factory=list, max_length=100)
    mode: StudySessionMode = StudySessionMode.DAILY_REVIEW
    limit: int = Field(default=20, ge=1, le=100)


class StudyQuestionResponse(BaseModel):
    id: uuid.UUID
    topic_id: uuid.UUID
    item_type: StudyItemType
    prompt: str
    options: list[str]
    difficulty: int
    reason: str


class StudySessionResponse(BaseModel):
    id: uuid.UUID
    course_id: uuid.UUID
    topic_id: uuid.UUID | None
    mode: StudySessionMode
    status: StudySessionStatus
    items: list[StudyQuestionResponse]
    answered_item_ids: list[uuid.UUID]
    completed_at: datetime | None
    created_at: datetime


class StudyAnswerRequest(BaseModel):
    item_id: uuid.UUID
    submitted_answer: str = Field(max_length=8000)
    rating: ReviewRating
    confidence: int = Field(ge=1, le=5)
    response_time_ms: int = Field(ge=0, le=86_400_000)
    hint_used: bool = False
    idempotency_key: str = Field(min_length=8, max_length=128)


class MemoryStateResponse(BaseModel):
    study_item_id: uuid.UUID
    half_life_days: float
    difficulty: int
    last_review_at: datetime
    next_review_at: datetime
    predicted_recall: float
    successful_reviews: int
    failed_reviews: int
    scheduler_version: str


class ReviewAttemptResponse(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    study_item_id: uuid.UUID
    submitted_answer: str
    expected_answer: str
    correct: bool
    score: float
    rating: ReviewRating
    confidence: int
    response_time_ms: int
    hint_used: bool
    grading_details: dict[str, Any]
    sources: list[SourcePassageResponse]
    reviewed_at: datetime
    overridden_correct: bool | None
    overridden_score: float | None
    override_reason: str | None
    memory: MemoryStateResponse


class ReviewOverrideRequest(BaseModel):
    correct: bool
    score: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=2000)


class ReviewQueueItemResponse(StudyQuestionResponse):
    due_at: datetime | None
    predicted_recall: float | None


class ReviewQueueResponse(BaseModel):
    course_id: uuid.UUID
    generated_at: datetime
    estimated_minutes: int
    items: list[ReviewQueueItemResponse]


class ProgressSummaryResponse(BaseModel):
    coverage: float
    mastery: float
    predicted_retention: float
    encountered_items: int
    total_items: int
    factors: dict[str, float | int | str]


class TopicProgressResponse(ProgressSummaryResponse):
    topic_id: uuid.UUID
    title: str


class CourseProgressResponse(ProgressSummaryResponse):
    course_id: uuid.UUID
    title: str
    topics: list[TopicProgressResponse]
    weakest_topic_id: uuid.UUID | None
    recommended_action: str


class ProgressSnapshotResponse(ProgressSummaryResponse):
    id: uuid.UUID
    captured_at: datetime


class ReviewObservationResponse(BaseModel):
    attempt_id: uuid.UUID
    reviewed_at: datetime
    score: float
    correct: bool
    overridden: bool


class RecallPredictionResponse(BaseModel):
    at: datetime
    predicted_recall: float


class ItemProgressResponse(BaseModel):
    study_item_id: uuid.UUID
    prompt: str
    mastery: float
    factors: dict[str, float | int]
    memory: MemoryStateResponse | None
    observations: list[ReviewObservationResponse]
    predictions: list[RecallPredictionResponse]


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

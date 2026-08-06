"""PostgreSQL persistence contracts and repositories."""

from note_rag.persistence.database import Database
from note_rag.persistence.models import (
    Base,
    CacheState,
    ChatMessageRecord,
    ChatRole,
    ChunkRecord,
    Conversation,
    Document,
    DocumentStatus,
    IndexingStatus,
    IngestionJob,
    IngestionJobStatus,
    QueryEmbeddingCache,
    RetrievalResultCache,
)
from note_rag.persistence.repositories import (
    ChatMessageRepository,
    ChunkRepository,
    ConversationRepository,
    DocumentRepository,
    IngestionJobRepository,
)

__all__ = [
    "Base",
    "CacheState",
    "ChatMessageRecord",
    "ChatMessageRepository",
    "ChatRole",
    "ChunkRecord",
    "ChunkRepository",
    "Conversation",
    "ConversationRepository",
    "Database",
    "Document",
    "DocumentRepository",
    "DocumentStatus",
    "IngestionJob",
    "IngestionJobRepository",
    "IngestionJobStatus",
    "IndexingStatus",
    "QueryEmbeddingCache",
    "RetrievalResultCache",
]

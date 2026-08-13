"""PostgreSQL persistence contracts and repositories."""

from note_rag.persistence.database import Database
from note_rag.persistence.models import (
    Base,
    CacheState,
    ChatMessageRecord,
    ChatRole,
    ChunkLexicalTerm,
    ChunkRecord,
    Conversation,
    ConversationMemoryRecord,
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
    ConversationMemoryRepository,
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
    "ChunkLexicalTerm",
    "ChunkRecord",
    "ChunkRepository",
    "Conversation",
    "ConversationMemoryRecord",
    "ConversationMemoryRepository",
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

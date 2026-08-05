"""PostgreSQL persistence contracts and repositories."""

from note_rag.persistence.database import Database
from note_rag.persistence.models import (
    Base,
    ChunkRecord,
    Document,
    DocumentStatus,
    IndexingStatus,
    IngestionJob,
    IngestionJobStatus,
)
from note_rag.persistence.repositories import (
    ChunkRepository,
    DocumentRepository,
    IngestionJobRepository,
)

__all__ = [
    "Base",
    "ChunkRecord",
    "ChunkRepository",
    "Database",
    "Document",
    "DocumentRepository",
    "DocumentStatus",
    "IngestionJob",
    "IngestionJobRepository",
    "IngestionJobStatus",
    "IndexingStatus",
]

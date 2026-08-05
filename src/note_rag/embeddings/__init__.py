"""Embedding providers and document indexing."""

from note_rag.embeddings.providers import (
    EmbeddingProvider,
    GeminiEmbeddingProvider,
)
from note_rag.embeddings.service import IndexingResult, IndexingService

__all__ = [
    "EmbeddingProvider",
    "IndexingResult",
    "IndexingService",
    "GeminiEmbeddingProvider",
]

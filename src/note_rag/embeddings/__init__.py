"""Embedding providers and document indexing."""

from note_rag.embeddings.providers import (
    DeterministicEmbeddingProvider,
    EmbeddingProvider,
    GeminiEmbeddingProvider,
    QueryEmbeddingProvider,
)
from note_rag.embeddings.service import IndexingResult, IndexingService

__all__ = [
    "EmbeddingProvider",
    "DeterministicEmbeddingProvider",
    "IndexingResult",
    "IndexingService",
    "GeminiEmbeddingProvider",
    "QueryEmbeddingProvider",
]

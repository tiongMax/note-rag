"""Search indexed chunks with dense and lexical signals."""

from note_rag.retrieval.cache import PersistentRetrievalCache
from note_rag.retrieval.models import (
    RetrievalHit,
    RetrievalResult,
    SearchFilters,
    SearchMode,
)
from note_rag.retrieval.repository import RankedChunk, RetrievalRepository
from note_rag.retrieval.service import RetrievalService

__all__ = [
    "PersistentRetrievalCache",
    "RankedChunk",
    "RetrievalHit",
    "RetrievalRepository",
    "RetrievalResult",
    "RetrievalService",
    "SearchFilters",
    "SearchMode",
]

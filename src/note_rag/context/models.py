"""Clean context-package contracts for later prompt construction."""

import uuid
from dataclasses import dataclass
from typing import Any

from note_rag.retrieval import SearchMode


@dataclass(frozen=True, slots=True)
class ContextChunk:
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
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class ContextPackage:
    query: str
    mode: SearchMode
    context: str
    chunks: list[ContextChunk]
    token_count: int
    token_budget: int
    candidates_considered: int
    duplicates_removed: int
    truncated: bool
    reranker_model: str | None

"""Framework-independent retrieval contracts."""

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SearchMode(StrEnum):
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class SearchFilters:
    document_ids: tuple[uuid.UUID, ...] = ()
    filenames: tuple[str, ...] = ()
    media_types: tuple[str, ...] = ()
    source_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    media_type: str
    position: int
    text: str
    source_metadata: dict[str, Any]
    score: float
    vector_score: float | None = None
    keyword_score: float | None = None


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query: str
    mode: SearchMode
    hits: list[RetrievalHit]

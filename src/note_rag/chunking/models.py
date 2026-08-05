"""Chunk contracts shared by ingestion, persistence, and retrieval phases."""

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Stable position metadata for a chunk in its source text."""

    index: int = Field(ge=0)
    token_start: int = Field(ge=0)
    token_end: int = Field(ge=0)
    token_count: int = Field(ge=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    source_id: str | None = None


class Chunk(BaseModel):
    """A source text slice and the metadata required by later phases."""

    text: str = Field(min_length=1)
    metadata: ChunkMetadata

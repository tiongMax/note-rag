"""Phase 1 HTTP request and response contracts."""

from pydantic import BaseModel, Field

from note_rag.chunking.models import Chunk


class ChunkTextRequest(BaseModel):
    text: str
    source_id: str | None = None
    chunk_size: int | None = Field(default=None, gt=0)
    chunk_overlap: int | None = Field(default=None, ge=0)


class ChunkTextResponse(BaseModel):
    token_count: int
    chunks: list[Chunk]

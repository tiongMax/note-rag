"""Token-aware text chunking."""

from typing import Protocol

from note_rag.chunking.models import Chunk, ChunkMetadata
from note_rag.chunking.tokens import RegexTokenCounter


class Chunker(Protocol):
    def chunk(self, text: str, *, source_id: str | None = None) -> list[Chunk]: ...


class TokenChunker:
    """Split text into deterministic token windows with optional overlap."""

    def __init__(
        self,
        chunk_size: int = 200,
        chunk_overlap: int = 20,
        token_counter: RegexTokenCounter | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.token_counter = token_counter or RegexTokenCounter()

    def chunk(self, text: str, *, source_id: str | None = None) -> list[Chunk]:
        """Return source-preserving chunks ordered by their token position."""

        tokens = self.token_counter.tokenize(text)
        if not tokens:
            return []

        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap

        for index, token_start in enumerate(range(0, len(tokens), step)):
            token_end = min(token_start + self.chunk_size, len(tokens))
            window = tokens[token_start:token_end]
            char_start = window[0].start
            char_end = window[-1].end
            chunks.append(
                Chunk(
                    text=text[char_start:char_end],
                    metadata=ChunkMetadata(
                        index=index,
                        token_start=token_start,
                        token_end=token_end,
                        token_count=len(window),
                        char_start=char_start,
                        char_end=char_end,
                        source_id=source_id,
                    ),
                )
            )
            if token_end == len(tokens):
                break

        return chunks

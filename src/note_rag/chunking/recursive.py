"""Structure-aware recursive chunking with token-bounded overlap."""

from __future__ import annotations

import bisect
import re
from collections.abc import Sequence

from note_rag.chunking.models import Chunk, ChunkMetadata
from note_rag.chunking.tokens import RegexTokenCounter, Token

_STRUCTURAL_SEPARATORS = (
    re.compile(r"(?m)(?=^#{1,6}[ \t]+\S)"),
    re.compile(r"\n[ \t]*\n+"),
    re.compile(r"(?<=[.!?])(?:[ \t]+|\n+)"),
    re.compile(r"\n+"),
)


class RecursiveChunker:
    """Prefer section, paragraph, sentence, and line boundaries."""

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
        tokens = self.token_counter.tokenize(text)
        if not tokens:
            return []

        novel_budget = self.chunk_size - self.chunk_overlap
        boundary_levels = self._boundary_levels(text, tokens)
        units = self._split_range(
            0,
            len(tokens),
            novel_budget,
            boundary_levels,
            level=0,
        )
        chunk_ranges = self._pack(
            units,
            self.chunk_size,
            self.chunk_overlap,
        )

        chunks = []
        for index, (token_start, token_end) in enumerate(chunk_ranges):
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
        return chunks

    @staticmethod
    def _boundary_levels(
        text: str,
        tokens: Sequence[Token],
    ) -> tuple[tuple[int, ...], ...]:
        token_starts = [token.start for token in tokens]
        levels = []
        for separator in _STRUCTURAL_SEPARATORS:
            boundaries = set()
            for match in separator.finditer(text):
                char_boundary = (
                    match.start()
                    if match.start() == match.end()
                    else match.end()
                )
                token_boundary = bisect.bisect_left(
                    token_starts,
                    char_boundary,
                )
                if 0 < token_boundary < len(tokens):
                    boundaries.add(token_boundary)
            levels.append(tuple(sorted(boundaries)))
        return tuple(levels)

    @classmethod
    def _split_range(
        cls,
        start: int,
        end: int,
        budget: int,
        boundary_levels: tuple[tuple[int, ...], ...],
        *,
        level: int,
    ) -> list[tuple[int, int]]:
        if end - start <= budget:
            return [(start, end)]
        if level >= len(boundary_levels):
            return [
                (position, min(position + budget, end))
                for position in range(start, end, budget)
            ]

        boundaries = [
            boundary
            for boundary in boundary_levels[level]
            if start < boundary < end
        ]
        if not boundaries:
            return cls._split_range(
                start,
                end,
                budget,
                boundary_levels,
                level=level + 1,
            )

        result = []
        points = [start, *boundaries, end]
        for piece_start, piece_end in zip(points, points[1:]):
            result.extend(
                cls._split_range(
                    piece_start,
                    piece_end,
                    budget,
                    boundary_levels,
                    level=level + 1,
                )
            )
        return result

    @staticmethod
    def _pack(
        units: Sequence[tuple[int, int]],
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[tuple[int, int]]:
        packed: list[tuple[int, int]] = []
        current_start, current_end = units[0]
        for unit_start, unit_end in units[1:]:
            if unit_start != current_end:
                raise AssertionError("recursive units must be contiguous")
            if unit_end - current_start <= chunk_size:
                current_end = unit_end
            else:
                packed.append((current_start, current_end))
                current_start = max(0, current_end - chunk_overlap)
                current_end = unit_end
        packed.append((current_start, current_end))
        return packed

"""Candidate reranking, deduplication, and token-budgeted context building."""

import json
import math
import re
from dataclasses import dataclass
from typing import Protocol

from note_rag.chunking import RegexTokenCounter
from note_rag.context.models import ContextChunk, ContextPackage
from note_rag.context.rerankers import Reranker
from note_rag.retrieval import (
    RetrievalHit,
    RetrievalResult,
    SearchFilters,
    SearchMode,
)

_WHITESPACE = re.compile(r"\s+")


class Retriever(Protocol):
    def search(
        self,
        query: str,
        *,
        mode: SearchMode = SearchMode.HYBRID,
        top_k: int = 10,
        vector_weight: float = 0.7,
        filters: SearchFilters | None = None,
    ) -> RetrievalResult: ...


@dataclass(frozen=True, slots=True)
class _ScoredHit:
    hit: RetrievalHit
    rerank_score: float | None
    score: float
    original_rank: int


class ContextBuilder:
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker | None = None,
        *,
        token_counter: RegexTokenCounter | None = None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.token_counter = token_counter or RegexTokenCounter()

    def build(
        self,
        query: str,
        *,
        mode: SearchMode = SearchMode.HYBRID,
        candidate_k: int = 20,
        max_chunks: int = 8,
        max_context_tokens: int = 1200,
        vector_weight: float = 0.7,
        rerank: bool = True,
        rerank_weight: float = 0.7,
        filters: SearchFilters | None = None,
    ) -> ContextPackage:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero")
        if max_chunks <= 0:
            raise ValueError("max_chunks must be greater than zero")
        if max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be greater than zero")
        if not 0.0 <= rerank_weight <= 1.0:
            raise ValueError("rerank_weight must be between zero and one")

        result = self.retriever.search(
            query,
            mode=mode,
            top_k=candidate_k,
            vector_weight=vector_weight,
            filters=filters,
        )
        scored = self._rerank(query, result.hits, rerank, rerank_weight)
        unique, duplicates_removed = self._deduplicate(scored)
        chunks, context = self._fit_budget(
            unique,
            max_chunks=max_chunks,
            max_context_tokens=max_context_tokens,
        )
        context_tokens = self.token_counter.count(context)
        truncated = (
            len(chunks) < len(unique)
            or any(chunk.truncated for chunk in chunks)
        )
        return ContextPackage(
            query=result.query,
            mode=result.mode,
            context=context,
            chunks=chunks,
            token_count=context_tokens,
            token_budget=max_context_tokens,
            candidates_considered=len(result.hits),
            duplicates_removed=duplicates_removed,
            truncated=truncated,
            reranker_model=(
                self.reranker.model_name
                if rerank and self.reranker is not None
                else None
            ),
        )

    def _rerank(
        self,
        query: str,
        hits: list[RetrievalHit],
        enabled: bool,
        weight: float,
    ) -> list[_ScoredHit]:
        if not enabled or self.reranker is None or not hits:
            return [
                _ScoredHit(hit, None, hit.score, rank)
                for rank, hit in enumerate(hits)
            ]

        rerank_scores = self.reranker.score(
            query,
            [hit.text for hit in hits],
        )
        if len(rerank_scores) != len(hits):
            raise ValueError("reranker returned the wrong score count")
        if not all(math.isfinite(score) for score in rerank_scores):
            raise ValueError("reranker returned a non-finite score")
        if not all(0.0 <= score <= 1.0 for score in rerank_scores):
            raise ValueError("reranker score must be between zero and one")

        retrieval_scores = self._normalize([hit.score for hit in hits])
        scored = [
            _ScoredHit(
                hit=hit,
                rerank_score=rerank_score,
                score=(
                    weight * rerank_score
                    + (1.0 - weight) * retrieval_score
                ),
                original_rank=rank,
            )
            for rank, (hit, retrieval_score, rerank_score) in enumerate(
                zip(hits, retrieval_scores, rerank_scores, strict=True)
            )
        ]
        scored.sort(
            key=lambda item: (
                -item.score,
                item.original_rank,
                str(item.hit.chunk_id),
            )
        )
        return scored

    @staticmethod
    def _normalize(scores: list[float]) -> list[float]:
        if not scores:
            return []
        low = min(scores)
        high = max(scores)
        if math.isclose(low, high):
            return [1.0] * len(scores)
        return [(score - low) / (high - low) for score in scores]

    @staticmethod
    def _deduplicate(
        hits: list[_ScoredHit],
    ) -> tuple[list[_ScoredHit], int]:
        seen_ids = set()
        seen_text = set()
        unique = []
        for item in hits:
            normalized = _WHITESPACE.sub(" ", item.hit.text).strip().casefold()
            if item.hit.chunk_id in seen_ids or normalized in seen_text:
                continue
            seen_ids.add(item.hit.chunk_id)
            seen_text.add(normalized)
            unique.append(item)
        return unique, len(hits) - len(unique)

    def _fit_budget(
        self,
        hits: list[_ScoredHit],
        *,
        max_chunks: int,
        max_context_tokens: int,
    ) -> tuple[list[ContextChunk], str]:
        chunks = []
        rendered = []
        for item in hits:
            if len(chunks) >= max_chunks:
                break
            citation_id = len(chunks) + 1
            separator = "\n\n" if rendered else ""
            header = self._header(citation_id, item.hit)
            prefix = f"{separator}{header}\n"
            prefix_tokens = self.token_counter.count(prefix)
            used_tokens = self.token_counter.count("".join(rendered))
            available = max_context_tokens - used_tokens - prefix_tokens
            if available <= 0:
                break

            text, text_tokens, was_truncated = self._truncate(
                item.hit.text,
                available,
            )
            if not text:
                break
            rendered.append(f"{prefix}{text}")
            chunks.append(
                ContextChunk(
                    citation_id=citation_id,
                    chunk_id=item.hit.chunk_id,
                    document_id=item.hit.document_id,
                    filename=item.hit.filename,
                    media_type=item.hit.media_type,
                    position=item.hit.position,
                    text=text,
                    token_count=text_tokens,
                    source_metadata=item.hit.source_metadata,
                    retrieval_score=item.hit.score,
                    rerank_score=item.rerank_score,
                    score=item.score,
                    truncated=was_truncated,
                )
            )
            if was_truncated:
                break
        return chunks, "".join(rendered)

    def _truncate(self, text: str, budget: int) -> tuple[str, int, bool]:
        tokens = self.token_counter.tokenize(text)
        if len(tokens) <= budget:
            return text, len(tokens), False
        if budget <= 0:
            return "", 0, True
        return text[: tokens[budget - 1].end], budget, True

    @staticmethod
    def _header(citation_id: int, hit: RetrievalHit) -> str:
        metadata = json.dumps(
            hit.source_metadata,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        return (
            f"[{citation_id}] Source: {hit.filename}; "
            f"media_type={hit.media_type}; position={hit.position}; "
            f"metadata={metadata}"
        )

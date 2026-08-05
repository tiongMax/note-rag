"""Vector, keyword, and hybrid retrieval orchestration."""

import math
from dataclasses import dataclass

from note_rag.embeddings import QueryEmbeddingProvider
from note_rag.persistence import Database
from note_rag.retrieval.models import (
    RetrievalHit,
    RetrievalResult,
    SearchFilters,
    SearchMode,
)
from note_rag.retrieval.repository import RankedChunk, RetrievalRepository


@dataclass(slots=True)
class _FusionEntry:
    candidate: RankedChunk
    score: float = 0.0
    vector_score: float | None = None
    keyword_score: float | None = None


class RetrievalService:
    def __init__(
        self,
        database: Database,
        embedding_provider: QueryEmbeddingProvider,
        *,
        candidate_multiplier: int = 4,
        rrf_k: int = 60,
    ) -> None:
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be greater than zero")
        if rrf_k < 0:
            raise ValueError("rrf_k cannot be negative")
        self.database = database
        self.embedding_provider = embedding_provider
        self.candidate_multiplier = candidate_multiplier
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        *,
        mode: SearchMode = SearchMode.HYBRID,
        top_k: int = 10,
        vector_weight: float = 0.7,
        filters: SearchFilters | None = None,
    ) -> RetrievalResult:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not 0.0 <= vector_weight <= 1.0:
            raise ValueError("vector_weight must be between zero and one")

        resolved_filters = filters or SearchFilters()
        candidate_limit = top_k * self.candidate_multiplier
        needs_vector = mode is SearchMode.VECTOR or (
            mode is SearchMode.HYBRID and vector_weight > 0
        )
        needs_keyword = mode is SearchMode.KEYWORD or (
            mode is SearchMode.HYBRID and vector_weight < 1
        )
        query_vector = self._embed_query(query) if needs_vector else None

        with self.database.session() as session:
            repository = RetrievalRepository(session)
            vector_hits = (
                repository.vector_search(
                    query_vector,
                    limit=candidate_limit,
                    filters=resolved_filters,
                )
                if query_vector is not None
                else []
            )
            keyword_hits = (
                repository.keyword_search(
                    query,
                    limit=candidate_limit,
                    filters=resolved_filters,
                )
                if needs_keyword
                else []
            )

        if mode is SearchMode.VECTOR:
            hits = [
                self._hit(item, score=item.score, vector_score=item.score)
                for item in vector_hits
            ]
        elif mode is SearchMode.KEYWORD:
            hits = [
                self._hit(item, score=item.score, keyword_score=item.score)
                for item in keyword_hits
            ]
        else:
            hits = self._fuse(vector_hits, keyword_hits, vector_weight)
        return RetrievalResult(query=query, mode=mode, hits=hits[:top_k])

    def _embed_query(self, query: str) -> list[float]:
        vector = self.embedding_provider.embed_query(query)
        if len(vector) != self.embedding_provider.dimension:
            raise ValueError("embedding provider returned the wrong dimension")
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("embedding vector contains a non-finite value")
        return vector

    def _fuse(
        self,
        vector_hits: list[RankedChunk],
        keyword_hits: list[RankedChunk],
        vector_weight: float,
    ) -> list[RetrievalHit]:
        by_id: dict[object, _FusionEntry] = {}
        for signal, weight, candidates in (
            ("vector", vector_weight, vector_hits),
            ("keyword", 1.0 - vector_weight, keyword_hits),
        ):
            for rank, candidate in enumerate(candidates, start=1):
                entry = by_id.setdefault(
                    candidate.chunk.id,
                    _FusionEntry(candidate),
                )
                entry.score += weight / (self.rrf_k + rank)
                if signal == "vector":
                    entry.vector_score = candidate.score
                else:
                    entry.keyword_score = candidate.score

        maximum = 1.0 / (self.rrf_k + 1)
        hits = [
            self._hit(
                entry.candidate,
                score=entry.score / maximum,
                vector_score=entry.vector_score,
                keyword_score=entry.keyword_score,
            )
            for entry in by_id.values()
        ]
        hits.sort(key=lambda item: (-item.score, str(item.chunk_id)))
        return hits

    @staticmethod
    def _hit(
        candidate: RankedChunk,
        *,
        score: float,
        vector_score: float | None = None,
        keyword_score: float | None = None,
    ) -> RetrievalHit:
        chunk = candidate.chunk
        return RetrievalHit(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            filename=candidate.filename,
            media_type=candidate.media_type,
            position=chunk.position,
            text=chunk.text,
            source_metadata=chunk.source_metadata,
            score=score,
            vector_score=vector_score,
            keyword_score=keyword_score,
        )

"""Vector, keyword, and hybrid retrieval orchestration."""

import math
from dataclasses import dataclass

from note_rag.embeddings import QueryEmbeddingProvider
from note_rag.persistence import Database
from note_rag.retrieval.cache import PersistentRetrievalCache
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
        cache: PersistentRetrievalCache | None = None,
    ) -> None:
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be greater than zero")
        if rrf_k < 0:
            raise ValueError("rrf_k cannot be negative")
        self.database = database
        self.embedding_provider = embedding_provider
        self.candidate_multiplier = candidate_multiplier
        self.rrf_k = rrf_k
        self.cache = cache

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
        cache_status = "disabled"
        corpus_version = 0
        cache_key = ""
        if self.cache is not None:
            cached, cache_status, corpus_version, cache_key = (
                self.cache.get_retrieval(
                    query=query,
                    mode=mode,
                    top_k=top_k,
                    vector_weight=vector_weight,
                    filters=resolved_filters,
                    model_name=self.embedding_provider.model_name,
                    candidate_multiplier=self.candidate_multiplier,
                    rrf_k=self.rrf_k,
                )
            )
            if cached is not None:
                return cached
        candidate_limit = top_k * self.candidate_multiplier
        needs_vector = mode is SearchMode.VECTOR or (
            mode is SearchMode.HYBRID and vector_weight > 0
        )
        needs_keyword = mode is SearchMode.KEYWORD or (
            mode is SearchMode.HYBRID and vector_weight < 1
        )
        embedding_cache_status = "not_used"
        query_vector = None
        if needs_vector:
            query_vector, embedding_cache_status = self._embed_query(query)

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
        result = RetrievalResult(
            query=query,
            mode=mode,
            hits=hits[:top_k],
            embedding_cache_status=embedding_cache_status,
            retrieval_cache_status=cache_status,
            corpus_version=corpus_version,
        )
        if self.cache is not None:
            self.cache.put_retrieval(
                cache_key,
                result,
                corpus_version=corpus_version,
            )
        return result

    def _embed_query(self, query: str) -> tuple[list[float], str]:
        cache_status = "disabled"
        if self.cache is not None:
            vector, cache_status = self.cache.get_embedding(
                model_name=self.embedding_provider.model_name,
                dimension=self.embedding_provider.dimension,
                query=query,
            )
            if vector is not None:
                return vector, cache_status
        try:
            vector = self.embedding_provider.embed_query(query)
        except Exception:
            if self.cache is not None:
                self.cache.record_provider_call("error")
            raise
        if self.cache is not None:
            self.cache.record_provider_call("success")
        if len(vector) != self.embedding_provider.dimension:
            raise ValueError("embedding provider returned the wrong dimension")
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("embedding vector contains a non-finite value")
        if self.cache is not None:
            self.cache.put_embedding(
                model_name=self.embedding_provider.model_name,
                dimension=self.embedding_provider.dimension,
                query=query,
                vector=vector,
            )
        return vector, cache_status

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
            token_count=chunk.token_count,
            token_start=chunk.token_start,
            token_end=chunk.token_end,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            source_metadata=chunk.source_metadata,
            score=score,
            vector_score=vector_score,
            keyword_score=keyword_score,
        )

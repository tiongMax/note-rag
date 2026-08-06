"""Vector, keyword, and hybrid retrieval orchestration."""

import math
import uuid
from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy import func, select

from note_rag.cache import PersistentCache, cache_key
from note_rag.embeddings import QueryEmbeddingProvider
from note_rag.persistence import ChunkRecord, Database, Document
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
        cache: PersistentCache | None = None,
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
        corpus_version = self._corpus_version()
        result_key = cache_key(
            {
                "version": 1,
                "corpus": corpus_version,
                "query": query.casefold(),
                "mode": mode.value,
                "top_k": top_k,
                "vector_weight": vector_weight,
                "filters": resolved_filters,
                "candidate_multiplier": self.candidate_multiplier,
                "rrf_k": self.rrf_k,
                "embedding_model": self.embedding_provider.model_name,
            }
        )
        if self.cache is not None:
            cached = self.cache.get("retrieval", result_key)
            if cached is not None:
                return self._deserialize_result(cached)
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
        result = RetrievalResult(query=query, mode=mode, hits=hits[:top_k])
        if self.cache is not None:
            self.cache.set("retrieval", result_key, self._serialize_result(result))
        return result

    def _embed_query(self, query: str) -> list[float]:
        key = cache_key(
            {
                "version": 1,
                "model": self.embedding_provider.model_name,
                "dimension": self.embedding_provider.dimension,
                "query": query.casefold(),
            }
        )
        cached = (
            self.cache.get("query_embedding", key)
            if self.cache is not None
            else None
        )
        vector = (
            [float(value) for value in cached]
            if cached is not None
            else self.embedding_provider.embed_query(query)
        )
        if len(vector) != self.embedding_provider.dimension:
            raise ValueError("embedding provider returned the wrong dimension")
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("embedding vector contains a non-finite value")
        if self.cache is not None and cached is None:
            self.cache.set("query_embedding", key, vector)
        return vector

    def _corpus_version(self) -> str:
        """Fingerprint indexed corpus state so stale retrieval entries are bypassed."""

        with self.database.session() as session:
            row = session.execute(
                select(
                    func.count(ChunkRecord.id),
                    func.max(ChunkRecord.updated_at),
                    func.count(Document.id.distinct()),
                    func.max(Document.updated_at),
                ).select_from(ChunkRecord).join(Document)
            ).one()
        return cache_key({"version": 1, "state": list(row)})

    @staticmethod
    def _serialize_result(result: RetrievalResult) -> dict[str, object]:
        return {
            "query": result.query,
            "mode": result.mode.value,
            "hits": [
                {
                    "chunk_id": str(hit.chunk_id),
                    "document_id": str(hit.document_id),
                    "filename": hit.filename,
                    "media_type": hit.media_type,
                    "position": hit.position,
                    "text": hit.text,
                    "source_metadata": hit.source_metadata,
                    "score": float(hit.score),
                    "vector_score": (
                        float(hit.vector_score)
                        if hit.vector_score is not None
                        else None
                    ),
                    "keyword_score": (
                        float(hit.keyword_score)
                        if hit.keyword_score is not None
                        else None
                    ),
                }
                for hit in result.hits
            ],
        }

    @staticmethod
    def _deserialize_result(payload: dict[str, object]) -> RetrievalResult:
        raw_hits = cast(list[dict[str, Any]], payload["hits"])
        hits = [
            RetrievalHit(
                chunk_id=uuid.UUID(item["chunk_id"]),
                document_id=uuid.UUID(item["document_id"]),
                filename=item["filename"],
                media_type=item["media_type"],
                position=item["position"],
                text=item["text"],
                source_metadata=item["source_metadata"],
                score=item["score"],
                vector_score=item["vector_score"],
                keyword_score=item["keyword_score"],
            )
            for item in raw_hits
        ]
        return RetrievalResult(
            query=str(payload["query"]),
            mode=SearchMode(str(payload["mode"])),
            hits=hits,
        )

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

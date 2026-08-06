"""Persistent exact caches for query embeddings and retrieval results."""

import hashlib
import json
import math
import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from sqlalchemy import delete
from sqlalchemy.orm import Session

from note_rag.persistence import (
    CacheState,
    Database,
    QueryEmbeddingCache,
    RetrievalResultCache,
)
from note_rag.retrieval.models import (
    RetrievalHit,
    RetrievalResult,
    SearchFilters,
    SearchMode,
)

_WHITESPACE = re.compile(r"\s+")
_CORPUS_VERSION_KEY = "corpus_version"


class CacheMetrics(Protocol):
    def record_cache_request(self, cache: str, result: str) -> None: ...

    def record_embedding_provider_call(self, result: str) -> None: ...

    def record_cache_invalidation(self, reason: str) -> None: ...

    def set_corpus_version(self, version: int) -> None: ...


class PersistentRetrievalCache:
    """Database-backed exact caches with TTL and corpus versioning."""

    def __init__(
        self,
        database: Database,
        *,
        enabled: bool = True,
        embedding_enabled: bool = True,
        retrieval_enabled: bool = True,
        embedding_ttl_seconds: int = 86_400,
        retrieval_ttl_seconds: int = 3_600,
        metrics: CacheMetrics | None = None,
    ) -> None:
        if embedding_ttl_seconds <= 0 or retrieval_ttl_seconds <= 0:
            raise ValueError("cache TTL values must be greater than zero")
        self.database = database
        self.enabled = enabled
        self.embedding_enabled = enabled and embedding_enabled
        self.retrieval_enabled = enabled and retrieval_enabled
        self.embedding_ttl = timedelta(seconds=embedding_ttl_seconds)
        self.retrieval_ttl = timedelta(seconds=retrieval_ttl_seconds)
        self.metrics = metrics

    @staticmethod
    def normalize_query(query: str) -> str:
        return _WHITESPACE.sub(" ", query.strip())

    def get_embedding(
        self,
        *,
        model_name: str,
        dimension: int,
        query: str,
    ) -> tuple[list[float] | None, str]:
        if not self.embedding_enabled:
            return None, "disabled"
        normalized = self.normalize_query(query)
        key_hash = _hash(
            {
                "version": 1,
                "model": model_name,
                "dimension": dimension,
                "query": normalized,
            }
        )
        with self.database.session() as session:
            entry = session.get(QueryEmbeddingCache, key_hash)
            if entry is None:
                self._cache_metric("embedding", "miss")
                return None, "miss"
            if _is_expired(entry.expires_at):
                session.delete(entry)
                self._cache_metric("embedding", "expired")
                return None, "expired"
            vector = [float(value) for value in entry.vector]
        if len(vector) != dimension or not all(math.isfinite(v) for v in vector):
            self._cache_metric("embedding", "invalid")
            return None, "invalid"
        self._cache_metric("embedding", "hit")
        return vector, "hit"

    def put_embedding(
        self,
        *,
        model_name: str,
        dimension: int,
        query: str,
        vector: list[float],
    ) -> None:
        if not self.embedding_enabled:
            return
        normalized = self.normalize_query(query)
        key_hash = _hash(
            {
                "version": 1,
                "model": model_name,
                "dimension": dimension,
                "query": normalized,
            }
        )
        expires_at = datetime.now(UTC) + self.embedding_ttl
        with self.database.session() as session:
            _upsert(
                session,
                QueryEmbeddingCache,
                {
                    "key_hash": key_hash,
                    "model_name": model_name,
                    "normalized_query": normalized,
                    "dimension": dimension,
                    "vector": [float(value) for value in vector],
                    "expires_at": expires_at,
                },
                update_columns=("vector", "expires_at", "updated_at"),
            )

    def get_retrieval(
        self,
        *,
        query: str,
        mode: SearchMode,
        top_k: int,
        vector_weight: float,
        filters: SearchFilters,
        model_name: str,
        candidate_multiplier: int,
        rrf_k: int,
    ) -> tuple[RetrievalResult | None, str, int, str]:
        corpus_version = self.corpus_version()
        key_hash = self.retrieval_key(
            query=query,
            mode=mode,
            top_k=top_k,
            vector_weight=vector_weight,
            filters=filters,
            model_name=model_name,
            candidate_multiplier=candidate_multiplier,
            rrf_k=rrf_k,
            corpus_version=corpus_version,
        )
        if not self.retrieval_enabled:
            return None, "disabled", corpus_version, key_hash
        with self.database.session() as session:
            entry = session.get(RetrievalResultCache, key_hash)
            if entry is None:
                self._cache_metric("retrieval", "miss")
                return None, "miss", corpus_version, key_hash
            if _is_expired(entry.expires_at):
                session.delete(entry)
                self._cache_metric("retrieval", "expired")
                return None, "expired", corpus_version, key_hash
            result = _deserialize_result(entry.payload)
        self._cache_metric("retrieval", "hit")
        return (
            RetrievalResult(
                query=query.strip(),
                mode=result.mode,
                hits=result.hits,
                embedding_cache_status="skipped",
                retrieval_cache_status="hit",
                corpus_version=corpus_version,
            ),
            "hit",
            corpus_version,
            key_hash,
        )

    def put_retrieval(
        self,
        key_hash: str,
        result: RetrievalResult,
        *,
        corpus_version: int,
    ) -> None:
        if not self.retrieval_enabled:
            return
        expires_at = datetime.now(UTC) + self.retrieval_ttl
        payload = _serialize_result(result)
        with self.database.session() as session:
            _upsert(
                session,
                RetrievalResultCache,
                {
                    "key_hash": key_hash,
                    "corpus_version": corpus_version,
                    "payload": payload,
                    "expires_at": expires_at,
                },
                update_columns=(
                    "corpus_version",
                    "payload",
                    "expires_at",
                    "updated_at",
                ),
            )

    def corpus_version(self) -> int:
        if not self.retrieval_enabled:
            return 0
        with self.database.session() as session:
            state = session.get(CacheState, _CORPUS_VERSION_KEY)
            version = state.value if state is not None else 1
        if self.metrics is not None:
            self.metrics.set_corpus_version(version)
        return version

    def invalidate_retrieval(
        self,
        *,
        reason: str,
        session: Session | None = None,
    ) -> int:
        if not self.retrieval_enabled:
            return 0
        if session is None:
            with self.database.session() as owned_session:
                version = self._invalidate(owned_session)
        else:
            version = self._invalidate(session)
        if self.metrics is not None:
            self.metrics.record_cache_invalidation(reason)
            self.metrics.set_corpus_version(version)
        return version

    @staticmethod
    def _invalidate(session: Session) -> int:
        state = session.get(CacheState, _CORPUS_VERSION_KEY)
        if state is None:
            state = CacheState(key=_CORPUS_VERSION_KEY, value=2)
            session.add(state)
        else:
            state.value += 1
        session.execute(delete(RetrievalResultCache))
        return state.value

    def retrieval_key(
        self,
        *,
        query: str,
        mode: SearchMode,
        top_k: int,
        vector_weight: float,
        filters: SearchFilters,
        model_name: str,
        candidate_multiplier: int,
        rrf_k: int,
        corpus_version: int,
    ) -> str:
        return _hash(
            {
                "version": 1,
                "query": self.normalize_query(query),
                "mode": mode.value,
                "top_k": top_k,
                "vector_weight": vector_weight,
                "filters": {
                    "document_ids": sorted(str(v) for v in filters.document_ids),
                    "filenames": sorted(filters.filenames),
                    "media_types": sorted(filters.media_types),
                    "source_metadata": filters.source_metadata,
                },
                "model": model_name,
                "candidate_multiplier": candidate_multiplier,
                "rrf_k": rrf_k,
                "corpus_version": corpus_version,
            }
        )

    def record_provider_call(self, result: str) -> None:
        if self.metrics is not None:
            self.metrics.record_embedding_provider_call(result)

    def _cache_metric(self, cache: str, result: str) -> None:
        if self.metrics is not None:
            self.metrics.record_cache_request(cache, result)


def _hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _is_expired(expires_at: datetime) -> bool:
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    return expires_at <= datetime.now(UTC)


def _upsert(
    session: Session,
    model: type[QueryEmbeddingCache] | type[RetrievalResultCache],
    values: dict[str, Any],
    *,
    update_columns: tuple[str, ...],
) -> None:
    dialect = session.bind.dialect.name if session.bind is not None else ""
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    elif dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    else:
        session.merge(model(**values))
        return
    statement = insert(model).values(**values)
    excluded = statement.excluded
    statement = statement.on_conflict_do_update(
        index_elements=["key_hash"],
        set_={
            column: (
                datetime.now(UTC)
                if column == "updated_at"
                else getattr(excluded, column)
            )
            for column in update_columns
        },
    )
    session.execute(statement)


def _serialize_result(result: RetrievalResult) -> dict[str, Any]:
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
                "token_count": hit.token_count,
                "token_start": hit.token_start,
                "token_end": hit.token_end,
                "char_start": hit.char_start,
                "char_end": hit.char_end,
                "source_metadata": hit.source_metadata,
                "score": float(hit.score),
                "vector_score": (
                    float(hit.vector_score) if hit.vector_score is not None else None
                ),
                "keyword_score": (
                    float(hit.keyword_score) if hit.keyword_score is not None else None
                ),
            }
            for hit in result.hits
        ],
    }


def _deserialize_result(payload: dict[str, Any]) -> RetrievalResult:
    return RetrievalResult(
        query=str(payload["query"]),
        mode=SearchMode(str(payload["mode"])),
        hits=[
            RetrievalHit(
                chunk_id=uuid.UUID(str(item["chunk_id"])),
                document_id=uuid.UUID(str(item["document_id"])),
                filename=str(item["filename"]),
                media_type=str(item["media_type"]),
                position=int(item["position"]),
                text=str(item["text"]),
                token_count=int(item["token_count"]),
                token_start=int(item["token_start"]),
                token_end=int(item["token_end"]),
                char_start=int(item["char_start"]),
                char_end=int(item["char_end"]),
                source_metadata=dict(item["source_metadata"]),
                score=float(item["score"]),
                vector_score=(
                    float(item["vector_score"])
                    if item["vector_score"] is not None
                    else None
                ),
                keyword_score=(
                    float(item["keyword_score"])
                    if item["keyword_score"] is not None
                    else None
                ),
            )
            for item in payload["hits"]
        ],
    )

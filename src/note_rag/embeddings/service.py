"""Batch embedding and pgvector indexing orchestration."""

import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy.orm import Session

from note_rag.embeddings.providers import EmbeddingProvider
from note_rag.persistence import (
    ChunkRepository,
    Database,
    DocumentRepository,
    IndexingStatus,
    IngestionJobRepository,
    IngestionJobStatus,
)


@dataclass(frozen=True, slots=True)
class IndexingResult:
    document_id: uuid.UUID
    status: IndexingStatus
    indexed_chunks: int
    embedding_model: str
    error_message: str | None = None


class RetrievalCacheInvalidator(Protocol):
    def invalidate_retrieval(
        self,
        *,
        reason: str,
        session: Session | None = None,
    ) -> int: ...


class IndexingService:
    def __init__(
        self,
        database: Database,
        provider: EmbeddingProvider,
        *,
        batch_size: int = 32,
        cache: RetrievalCacheInvalidator | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.database = database
        self.provider = provider
        self.batch_size = batch_size
        self.cache = cache

    def index_document(
        self,
        document_id: uuid.UUID,
        *,
        job_id: uuid.UUID | None = None,
        force: bool = False,
    ) -> IndexingResult:
        with self.database.session() as session:
            document = DocumentRepository(session).get(document_id)
            if document is None:
                raise LookupError("document not found")
            chunks = ChunkRepository(session).list_for_document(document_id)
            if (
                not force
                and document.indexing_status is IndexingStatus.INDEXED
                and all(chunk.embedding is not None for chunk in chunks)
            ):
                return self._result(document_id, document, len(chunks))

            job = (
                IngestionJobRepository(session).get(job_id)
                if job_id is not None
                else None
            )
            document.indexing_status = IndexingStatus.INDEXING
            document.indexing_error = None
            if job is not None:
                IngestionJobRepository(session).set_status(
                    job, IngestionJobStatus.EMBEDDING, progress=70
                )
            try:
                indexed = 0
                for start in range(0, len(chunks), self.batch_size):
                    batch = chunks[start : start + self.batch_size]
                    vectors = self.provider.embed([chunk.text for chunk in batch])
                    self._validate_vectors(vectors, len(batch))
                    now = datetime.now(UTC)
                    for chunk, vector in zip(batch, vectors, strict=True):
                        chunk.embedding = vector
                        chunk.embedding_model = self.provider.model_name
                        chunk.embedded_at = now
                        indexed += 1
                if job is not None:
                    IngestionJobRepository(session).set_status(
                        job, IngestionJobStatus.INDEXING, progress=90
                    )
                document.indexing_status = IndexingStatus.INDEXED
                document.embedding_model = self.provider.model_name
                document.indexed_at = datetime.now(UTC)
                if self.cache is not None:
                    self.cache.invalidate_retrieval(
                        reason="document_indexed",
                        session=session,
                    )
                if job is not None:
                    IngestionJobRepository(session).set_status(
                        job, IngestionJobStatus.INDEXING, progress=95
                    )
                return self._result(document_id, document, indexed)
            except Exception as error:
                message = str(error) or error.__class__.__name__
                document.indexing_status = IndexingStatus.FAILED
                document.indexing_error = message
                if job is not None:
                    IngestionJobRepository(session).set_status(
                        job,
                        IngestionJobStatus.FAILED,
                        error_message=message,
                    )
                return self._result(document_id, document, 0)

    def _validate_vectors(
        self, vectors: list[list[float]], expected_count: int
    ) -> None:
        if len(vectors) != expected_count:
            raise ValueError("embedding provider returned the wrong vector count")
        for vector in vectors:
            if len(vector) != self.provider.dimension:
                raise ValueError("embedding provider returned the wrong dimension")
            if not all(math.isfinite(value) for value in vector):
                raise ValueError("embedding vector contains a non-finite value")

    def _result(self, document_id, document, indexed: int) -> IndexingResult:
        return IndexingResult(
            document_id=document_id,
            status=document.indexing_status,
            indexed_chunks=indexed,
            embedding_model=self.provider.model_name,
            error_message=document.indexing_error,
        )

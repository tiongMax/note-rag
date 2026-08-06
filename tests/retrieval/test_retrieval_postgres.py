import os
import uuid
from pathlib import Path

import pytest

from note_rag.embeddings import IndexingService
from note_rag.ingest import IngestionPipeline, LocalFileStorage
from note_rag.persistence import Database, DocumentRepository
from note_rag.persistence.settings import DatabaseSettings
from note_rag.retrieval import PersistentRetrievalCache, RetrievalService, SearchFilters

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")


class FakeEmbeddingProvider:
    model_name = "fake-768"
    dimension = 768

    def __init__(self) -> None:
        self.query_calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, *([0.0] * 767)] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        self.query_calls += 1
        return [1.0, *([0.0] * 767)]


@pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL is required for the PostgreSQL integration test",
)
def test_live_hybrid_retrieval_with_filters(tmp_path: Path) -> None:
    assert TEST_DATABASE_URL is not None
    database = Database(DatabaseSettings(url=TEST_DATABASE_URL))
    marker = f"retrievalmarker{uuid.uuid4().hex}"
    result = None
    cache = PersistentRetrievalCache(database)
    try:
        result = IngestionPipeline(
            database,
            LocalFileStorage(tmp_path),
        ).ingest(
            filename="phase-five-smoke.txt",
            media_type="text/plain",
            content=f"{marker} hybrid retrieval".encode(),
        )
        IndexingService(database, FakeEmbeddingProvider()).index_document(
            result.document_id
        )

        provider = FakeEmbeddingProvider()
        retrieval = RetrievalService(database, provider, cache=cache)
        search = retrieval.search(
            marker,
            filters=SearchFilters(
                document_ids=(result.document_id,),
                source_metadata={"source_id": "phase-five-smoke.txt"},
            ),
        )
        warm = retrieval.search(
            marker,
            filters=SearchFilters(
                document_ids=(result.document_id,),
                source_metadata={"source_id": "phase-five-smoke.txt"},
            ),
        )

        assert len(search.hits) == 1
        assert search.hits[0].document_id == result.document_id
        assert search.hits[0].vector_score is not None
        assert search.hits[0].keyword_score is not None
        assert warm.retrieval_cache_status == "hit"
        assert provider.query_calls == 1
    finally:
        if result is not None:
            with database.session() as session:
                document = DocumentRepository(session).get(result.document_id)
                if document is not None:
                    DocumentRepository(session).delete(document)
                    cache.invalidate_retrieval(
                        reason="test_cleanup",
                        session=session,
                    )
        database.dispose()

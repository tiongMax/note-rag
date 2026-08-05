import os
import uuid
from pathlib import Path

import pytest

from note_rag.embeddings import IndexingService
from note_rag.ingest import IngestionPipeline, LocalFileStorage
from note_rag.persistence import (
    ChunkRepository,
    Database,
    DocumentRepository,
    IndexingStatus,
)
from note_rag.persistence.settings import DatabaseSettings

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")


class FakeEmbeddingProvider:
    model_name = "fake-768"
    dimension = 768

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, *([0.0] * 767)] for _ in texts]


@pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL is required for the PostgreSQL integration test",
)
def test_live_pgvector_round_trip(tmp_path: Path) -> None:
    assert TEST_DATABASE_URL is not None
    database = Database(DatabaseSettings(url=TEST_DATABASE_URL))
    content = f"phase four vector smoke test {uuid.uuid4()}".encode()
    result = None
    try:
        result = IngestionPipeline(
            database,
            LocalFileStorage(tmp_path),
        ).ingest(
            filename="vector-smoke.txt",
            media_type="text/plain",
            content=content,
        )
        indexed = IndexingService(
            database,
            FakeEmbeddingProvider(),
        ).index_document(result.document_id)

        assert indexed.status is IndexingStatus.INDEXED
        with database.session() as session:
            chunks = ChunkRepository(session).list_for_document(result.document_id)
            assert chunks[0].embedding is not None
            assert len(chunks[0].embedding) == 768
    finally:
        if result is not None:
            with database.session() as session:
                document = DocumentRepository(session).get(result.document_id)
                if document is not None:
                    DocumentRepository(session).delete(document)
        database.dispose()

import hashlib
import os
import uuid
from pathlib import Path

import pytest

from note_rag.chunking import TokenChunker
from note_rag.ingest import IngestionPipeline, LocalFileStorage
from note_rag.persistence import Database, DocumentRepository, DocumentStatus
from note_rag.persistence.settings import DatabaseSettings

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")


@pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL is required for the PostgreSQL integration test",
)
def test_live_postgres_ingestion_round_trip(tmp_path: Path) -> None:
    assert TEST_DATABASE_URL is not None
    database = Database(DatabaseSettings(url=TEST_DATABASE_URL))
    content = f"phase three smoke test {uuid.uuid4()}".encode()
    content_hash = hashlib.sha256(content).hexdigest()
    pipeline = IngestionPipeline(
        database,
        LocalFileStorage(tmp_path),
        chunker=TokenChunker(chunk_size=3, chunk_overlap=1),
    )
    try:
        result = pipeline.ingest(
            filename="phase-3-smoke.txt",
            media_type="text/plain",
            content=content,
        )

        assert result.status is DocumentStatus.READY
        assert result.chunk_count > 0
        with database.session() as session:
            stored = DocumentRepository(session).get(result.document_id)
            assert stored is not None
            assert stored.content_hash == content_hash
    finally:
        with database.session() as session:
            stored = DocumentRepository(session).get_by_content_hash(content_hash)
            if stored is not None:
                DocumentRepository(session).delete(stored)
        database.dispose()

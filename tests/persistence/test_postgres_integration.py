import os

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session

from note_rag.chunking import TokenChunker
from note_rag.persistence import ChunkRepository, Document, DocumentRepository

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")


@pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL is required for the PostgreSQL integration test",
)
def test_migration_and_repository_round_trip_on_postgres() -> None:
    assert TEST_DATABASE_URL is not None
    engine = create_engine(TEST_DATABASE_URL, pool_pre_ping=True)
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    try:
        assert {"documents", "chunks", "ingestion_jobs"} <= set(
            inspect(connection).get_table_names()
        )
        document = DocumentRepository(session).add(
            Document(filename="phase-2-smoke.txt", media_type="text/plain")
        )
        generated = TokenChunker(chunk_size=3, chunk_overlap=1).chunk(
            "one two three four",
            source_id=document.filename,
        )
        stored = ChunkRepository(session).add_from_chunks(document, generated)

        session.expire_all()

        assert DocumentRepository(session).get(document.id) is not None
        assert [chunk.text for chunk in stored] == [
            "one two three",
            "three four",
        ]
    finally:
        session.close()
        transaction.rollback()
        connection.close()
        engine.dispose()

from pathlib import Path

from note_rag.chunking import TokenChunker
from note_rag.ingest import IngestionPipeline, LocalFileStorage
from note_rag.persistence import (
    ChunkRepository,
    Database,
    DocumentRepository,
    DocumentStatus,
    IngestionJobRepository,
    IngestionJobStatus,
)


def build_pipeline(database: Database, tmp_path: Path) -> IngestionPipeline:
    return IngestionPipeline(
        database,
        LocalFileStorage(tmp_path),
        chunker=TokenChunker(chunk_size=3, chunk_overlap=1),
    )


def test_ingests_text_into_persisted_chunks(
    database: Database,
    tmp_path: Path,
) -> None:
    result = build_pipeline(database, tmp_path).ingest(
        filename="lesson.txt",
        media_type="text/plain",
        content=b"zero one two three four",
    )

    assert result.status is DocumentStatus.READY
    assert result.duplicate is False
    assert result.chunk_count == 2
    assert result.token_count == 5
    with database.session() as session:
        document = DocumentRepository(session).get(result.document_id)
        assert document is not None
        assert document.storage_uri is not None
        chunks = ChunkRepository(session).list_for_document(result.document_id)
        assert [chunk.text for chunk in chunks] == [
            "zero one two",
            "two three four",
        ]
        assert chunks[0].source_metadata == {
            "source_id": "lesson.txt",
            "format": "text",
        }
        assert result.job_id is not None
        job = IngestionJobRepository(session).get(result.job_id)
        assert job is not None
        assert job.status is IngestionJobStatus.COMPLETED
        assert job.progress == 100


def test_detects_duplicate_content_before_storing_again(
    database: Database,
    tmp_path: Path,
) -> None:
    pipeline = build_pipeline(database, tmp_path)
    first = pipeline.ingest(
        filename="first.txt",
        media_type="text/plain",
        content=b"identical content",
    )
    duplicate = pipeline.ingest(
        filename="renamed.txt",
        media_type="text/plain",
        content=b"identical content",
    )

    assert duplicate.duplicate is True
    assert duplicate.document_id == first.document_id
    with database.session() as session:
        assert len(DocumentRepository(session).list()) == 1


def test_persists_failed_parse_state(
    database: Database,
    tmp_path: Path,
) -> None:
    result = build_pipeline(database, tmp_path).ingest(
        filename="broken.txt",
        media_type="text/plain",
        content=b"\xff\xfe\xfa",
    )

    assert result.status is DocumentStatus.FAILED
    assert "valid UTF-8" in (result.error_message or "")
    with database.session() as session:
        document = DocumentRepository(session).get(result.document_id)
        assert document is not None
        assert document.status is DocumentStatus.FAILED
        assert ChunkRepository(session).list_for_document(document.id) == []
        assert result.job_id is not None
        job = IngestionJobRepository(session).get(result.job_id)
        assert job is not None
        assert job.status is IngestionJobStatus.FAILED

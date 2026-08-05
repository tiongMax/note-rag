from pathlib import Path

from note_rag.chunking import TokenChunker
from note_rag.embeddings import IndexingService
from note_rag.ingest import IngestionPipeline, LocalFileStorage
from note_rag.persistence import (
    ChunkRepository,
    Database,
    IndexingStatus,
)


class FakeEmbeddingProvider:
    model_name = "fake-768"
    dimension = 768

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text)), *([0.0] * 767)] for text in texts]


def test_batches_indexes_and_reindexes_document(
    database: Database,
    tmp_path: Path,
) -> None:
    ingestion = IngestionPipeline(
        database,
        LocalFileStorage(tmp_path),
        chunker=TokenChunker(chunk_size=2, chunk_overlap=0),
    ).ingest(
        filename="notes.txt",
        media_type="text/plain",
        content=b"one two three four five",
    )
    provider = FakeEmbeddingProvider()
    service = IndexingService(database, provider, batch_size=2)

    indexed = service.index_document(ingestion.document_id)
    reindexed = service.index_document(ingestion.document_id, force=True)

    assert indexed.status is IndexingStatus.INDEXED
    assert indexed.indexed_chunks == 3
    assert reindexed.status is IndexingStatus.INDEXED
    assert [len(batch) for batch in provider.calls] == [2, 1, 2, 1]
    with database.session() as session:
        chunks = ChunkRepository(session).list_for_document(ingestion.document_id)
        assert all(chunk.embedding is not None for chunk in chunks)
        assert all(chunk.embedding_model == "fake-768" for chunk in chunks)


def test_records_provider_failure(database: Database, tmp_path: Path) -> None:
    ingestion = IngestionPipeline(
        database,
        LocalFileStorage(tmp_path),
    ).ingest(
        filename="notes.txt",
        media_type="text/plain",
        content=b"some content",
    )

    class BrokenProvider(FakeEmbeddingProvider):
        def embed(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("model unavailable")

    result = IndexingService(database, BrokenProvider()).index_document(
        ingestion.document_id
    )

    assert result.status is IndexingStatus.FAILED
    assert result.error_message == "model unavailable"

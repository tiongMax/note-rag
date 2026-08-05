from note_rag.persistence import ChunkRecord, Database, Document, DocumentRepository
from note_rag.retrieval import RetrievalService, SearchFilters, SearchMode


class QueryEmbeddingProvider:
    model_name = "fake-768"
    dimension = 768

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, *([0.0] * 767)] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [1.0, *([0.0] * 767)]


def add_chunk(
    database: Database,
    *,
    filename: str,
    media_type: str,
    text: str,
    embedding: list[float],
    metadata: dict[str, object],
) -> None:
    with database.session() as session:
        document = DocumentRepository(session).add(
            Document(filename=filename, media_type=media_type)
        )
        session.add(
            ChunkRecord(
                document=document,
                position=0,
                text=text,
                token_count=2,
                token_start=0,
                token_end=2,
                char_start=0,
                char_end=len(text),
                source_metadata=metadata,
                embedding=embedding,
            )
        )


def test_vector_keyword_and_hybrid_search(database: Database) -> None:
    add_chunk(
        database,
        filename="vectors.txt",
        media_type="text/plain",
        text="dense concepts",
        embedding=[1.0, *([0.0] * 767)],
        metadata={"page": 1},
    )
    add_chunk(
        database,
        filename="words.md",
        media_type="text/markdown",
        text="orchard apple",
        embedding=[0.0, 1.0, *([0.0] * 766)],
        metadata={"page": 2},
    )
    service = RetrievalService(database, QueryEmbeddingProvider())

    vector = service.search("apple", mode=SearchMode.VECTOR)
    keyword = service.search("apple", mode=SearchMode.KEYWORD)
    hybrid = service.search("apple", mode=SearchMode.HYBRID, vector_weight=0.5)

    assert vector.hits[0].filename == "vectors.txt"
    assert vector.hits[0].vector_score == 1.0
    assert keyword.hits[0].filename == "words.md"
    assert keyword.hits[0].keyword_score is not None
    assert {hit.filename for hit in hybrid.hits} == {"vectors.txt", "words.md"}
    assert all(0.0 <= hit.score <= 1.0 for hit in hybrid.hits)


def test_filters_apply_to_all_search_modes(database: Database) -> None:
    for filename, page in (("one.txt", 1), ("two.txt", 2)):
        add_chunk(
            database,
            filename=filename,
            media_type="text/plain",
            text="shared keyword",
            embedding=[1.0, *([0.0] * 767)],
            metadata={"page": page, "kind": "lesson"},
        )
    service = RetrievalService(database, QueryEmbeddingProvider())

    result = service.search(
        "shared",
        filters=SearchFilters(
            filenames=("two.txt",),
            media_types=("text/plain",),
            source_metadata={"page": 2},
        ),
    )

    assert [hit.filename for hit in result.hits] == ["two.txt"]
    assert result.hits[0].source_metadata["kind"] == "lesson"


def test_keyword_mode_does_not_embed(database: Database) -> None:
    class FailingProvider(QueryEmbeddingProvider):
        def embed_query(self, query: str) -> list[float]:
            raise AssertionError("keyword search should not embed")

    add_chunk(
        database,
        filename="keyword.txt",
        media_type="text/plain",
        text="lexical match",
        embedding=[1.0, *([0.0] * 767)],
        metadata={},
    )

    result = RetrievalService(database, FailingProvider()).search(
        "lexical",
        mode=SearchMode.KEYWORD,
    )

    assert len(result.hits) == 1

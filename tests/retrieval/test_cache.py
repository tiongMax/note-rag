from fastapi.testclient import TestClient
from sqlalchemy import func, select

from note_rag.api.app import create_app
from note_rag.api.observability import MetricsRegistry
from note_rag.api.settings import ApiSettings
from note_rag.persistence import (
    Database,
    QueryEmbeddingCache,
    RetrievalResultCache,
)
from note_rag.retrieval import (
    PersistentRetrievalCache,
    RetrievalService,
    SearchFilters,
    SearchMode,
)
from tests.retrieval.test_service import QueryEmbeddingProvider, add_chunk


class CountingProvider(QueryEmbeddingProvider):
    def __init__(self) -> None:
        self.query_calls = 0

    def embed_query(self, query: str) -> list[float]:
        self.query_calls += 1
        return super().embed_query(query)


def build_service(
    database: Database,
) -> tuple[RetrievalService, CountingProvider, PersistentRetrievalCache]:
    provider = CountingProvider()
    cache = PersistentRetrievalCache(
        database,
        metrics=MetricsRegistry(),
    )
    return RetrievalService(database, provider, cache=cache), provider, cache


def test_repeated_query_skips_provider_and_retrieval(database: Database) -> None:
    add_chunk(
        database,
        filename="cache.txt",
        media_type="text/plain",
        text="cached apple",
        embedding=[1.0, *([0.0] * 767)],
        metadata={},
    )
    service, provider, _cache = build_service(database)

    cold = service.search("  apple  ", mode=SearchMode.HYBRID)
    warm = service.search("apple", mode=SearchMode.HYBRID)

    assert cold.embedding_cache_status == "miss"
    assert cold.retrieval_cache_status == "miss"
    assert warm.embedding_cache_status == "skipped"
    assert warm.retrieval_cache_status == "hit"
    assert warm.query == "apple"
    assert provider.query_calls == 1

    with database.session() as session:
        assert (
            session.scalar(select(func.count()).select_from(QueryEmbeddingCache)) == 1
        )
        assert (
            session.scalar(select(func.count()).select_from(RetrievalResultCache)) == 1
        )


def test_filters_isolate_retrieval_but_reuse_embedding(database: Database) -> None:
    for filename in ("one.txt", "two.txt"):
        add_chunk(
            database,
            filename=filename,
            media_type="text/plain",
            text="shared apple",
            embedding=[1.0, *([0.0] * 767)],
            metadata={},
        )
    service, provider, _cache = build_service(database)

    first = service.search(
        "apple",
        filters=SearchFilters(filenames=("one.txt",)),
    )
    second = service.search(
        "apple",
        filters=SearchFilters(filenames=("two.txt",)),
    )

    assert first.retrieval_cache_status == "miss"
    assert second.retrieval_cache_status == "miss"
    assert second.embedding_cache_status == "hit"
    assert [hit.filename for hit in second.hits] == ["two.txt"]
    assert provider.query_calls == 1


def test_corpus_invalidation_preserves_embedding_cache(database: Database) -> None:
    add_chunk(
        database,
        filename="versioned.txt",
        media_type="text/plain",
        text="versioned apple",
        embedding=[1.0, *([0.0] * 767)],
        metadata={},
    )
    service, provider, cache = build_service(database)
    initial = service.search("apple")
    assert service.search("apple").retrieval_cache_status == "hit"

    new_version = cache.invalidate_retrieval(reason="test_update")
    after_update = service.search("apple")

    assert new_version > initial.corpus_version
    assert after_update.corpus_version == new_version
    assert after_update.retrieval_cache_status == "miss"
    assert after_update.embedding_cache_status == "hit"
    assert provider.query_calls == 1


def test_cache_layers_can_be_enabled_independently(database: Database) -> None:
    add_chunk(
        database,
        filename="embedding-only.txt",
        media_type="text/plain",
        text="embedding only apple",
        embedding=[1.0, *([0.0] * 767)],
        metadata={},
    )
    provider = CountingProvider()
    cache = PersistentRetrievalCache(
        database,
        embedding_enabled=True,
        retrieval_enabled=False,
    )
    service = RetrievalService(database, provider, cache=cache)

    first = service.search("apple")
    second = service.search("apple")

    assert first.retrieval_cache_status == "disabled"
    assert second.retrieval_cache_status == "disabled"
    assert second.embedding_cache_status == "hit"
    assert provider.query_calls == 1


def test_search_exposes_cache_diagnostics_and_metrics(database: Database) -> None:
    add_chunk(
        database,
        filename="api-cache.txt",
        media_type="text/plain",
        text="observable apple",
        embedding=[1.0, *([0.0] * 767)],
        metadata={},
    )
    provider = CountingProvider()
    app = create_app(
        ApiSettings(background_worker_enabled=False),
        database=database,
        embedding_provider=provider,
    )
    client = TestClient(app)
    payload = {"query": "apple", "mode": "hybrid"}

    cold = client.post("/api/v1/retrieval/search", json=payload)
    warm = client.post("/api/v1/retrieval/search", json=payload)
    metrics = client.get("/metrics")

    assert cold.headers["x-embedding-cache"] == "miss"
    assert cold.headers["x-retrieval-cache"] == "miss"
    assert warm.headers["x-embedding-cache"] == "skipped"
    assert warm.headers["x-retrieval-cache"] == "hit"
    assert warm.headers["x-corpus-version"] == "1"
    assert (
        'note_rag_cache_requests_total{cache="retrieval",result="hit"} 1'
        in metrics.text
    )
    assert 'note_rag_embedding_provider_calls_total{result="success"} 1' in metrics.text

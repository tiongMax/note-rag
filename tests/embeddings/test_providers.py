from note_rag.embeddings import DeterministicEmbeddingProvider


def test_deterministic_provider_returns_stable_vectors() -> None:
    provider = DeterministicEmbeddingProvider(dimension=8, delay_ms=0)

    first = provider.embed_query("same query")
    second = provider.embed_query("same query")
    different = provider.embed_query("different query")

    assert first == second
    assert first != different
    assert len(first) == 8
    assert provider.model_name == "benchmark-deterministic-8"

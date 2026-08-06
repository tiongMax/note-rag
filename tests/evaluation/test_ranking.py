import pytest

from note_rag.evaluation import Bm25Index, weighted_rrf


def hit(identifier: str, text: str) -> dict[str, object]:
    return {"chunk_id": identifier, "text": text}


def test_bm25_rewards_rare_exact_terms_and_is_deterministic() -> None:
    index = Bm25Index(
        [
            hit("a", "scheduler scheduler process"),
            hit("b", "mutex semaphore synchronization"),
            hit("c", "process state transition"),
        ]
    )

    results = index.search("scheduler process", limit=3)

    assert [result["chunk_id"] for result in results] == ["a", "c"]
    assert results[0]["bm25_score"] > results[1]["bm25_score"]


def test_bm25_validates_parameters() -> None:
    with pytest.raises(ValueError, match="k1"):
        Bm25Index([], k1=0)
    with pytest.raises(ValueError, match="b"):
        Bm25Index([], b=2)


def test_weighted_rrf_combines_rankings_and_preserves_signal_scores() -> None:
    dense = [
        {**hit("a", "alpha"), "vector_score": 0.9},
        {**hit("b", "beta"), "vector_score": 0.8},
    ]
    bm25 = [
        {**hit("b", "beta"), "bm25_score": 4.0},
        {**hit("c", "gamma"), "bm25_score": 3.0},
    ]

    results = weighted_rrf(((dense, 0.5), (bm25, 0.5)), rrf_k=60)

    assert results[0]["chunk_id"] == "b"
    assert results[0]["vector_score"] == 0.8
    assert results[0]["bm25_score"] == 4.0

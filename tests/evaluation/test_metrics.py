import pytest

from note_rag.evaluation import (
    GoldPassage,
    RetrievedChunk,
    aggregate_query_metrics,
    evaluate_query,
    passage_coverage,
)


def passage(start: int = 100, end: int = 200) -> GoldPassage:
    return GoldPassage("document.pdf", start, end)


def chunk(start: int, end: int) -> RetrievedChunk:
    return RetrievedChunk("document.pdf", start, end)


def test_passage_coverage_uses_source_and_character_intersection() -> None:
    assert passage_coverage(chunk(50, 150), passage()) == pytest.approx(0.5)
    assert passage_coverage(chunk(100, 200), passage()) == pytest.approx(1.0)
    assert (
        passage_coverage(
            RetrievedChunk("other.pdf", 100, 200),
            passage(),
        )
        == 0
    )


def test_evaluates_ranking_and_combined_passage_coverage() -> None:
    chunks = [
        RetrievedChunk("other.pdf", 0, 100),
        chunk(100, 140),
        chunk(140, 200),
    ]

    metrics = evaluate_query(
        [passage()],
        chunks,
        relevance_threshold=0.5,
    )

    assert metrics["mrr"] == pytest.approx(1 / 3)
    assert metrics["precision_at_5"] == pytest.approx(1 / 5)
    assert metrics["irrelevant_at_5"] == 4
    assert metrics["recall_at_5"] == pytest.approx(1.0)
    assert metrics["recall_at_20"] == pytest.approx(1.0)
    assert metrics["passage_coverage_at_5"] == pytest.approx(1.0)
    assert metrics["boundary_completeness_at_5"] == 0
    assert 0 < metrics["ndcg_at_10"] < 1


def test_boundary_completeness_requires_one_containing_chunk() -> None:
    metrics = evaluate_query([passage()], [chunk(50, 250)])

    assert metrics["boundary_completeness_at_5"] == 1
    assert metrics["boundary_completeness_at_10"] == 1
    assert metrics["recall_at_5"] == 1


def test_macro_aggregates_metrics_and_latency_percentiles() -> None:
    summary = aggregate_query_metrics(
        [{"mrr": 1.0}, {"mrr": 0.0}],
        [10.0, 20.0, 30.0],
    )

    assert summary["query_count"] == 2
    assert summary["request_count"] == 3
    assert summary["mrr"] == pytest.approx(0.5)
    assert summary["latency_mean_ms"] == pytest.approx(20)
    assert summary["latency_p50_ms"] == pytest.approx(20)
    assert summary["latency_p95_ms"] == pytest.approx(29)

import pytest

from note_rag.evaluation import (
    GoldPassage,
    RetrievedChunk,
    aggregate_query_metrics,
    evaluate_query,
    passage_coverage,
)
from note_rag.evaluation.metrics import evaluate_trace, token_f1
from note_rag.evaluation.models import BenchmarkCase, EvaluationTrace, RetrievedContext


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


def _case() -> BenchmarkCase:
    return BenchmarkCase.model_validate(
        {
            "id": "q1",
            "question": "What and who?",
            "answerable": True,
            "reference_answer": "Version 2 launched and Alex approved it.",
            "evidence": [
                {
                    "document": "release.md",
                    "quote": "Version 2 launched.",
                    "group": "release",
                    "relevance": 3,
                },
                {
                    "document": "approval.md",
                    "quote": "Alex approved it.",
                    "group": "approval",
                    "relevance": 2,
                },
            ],
        }
    )


def _chunk(
    citation_id: int,
    filename: str,
    text: str,
) -> RetrievedContext:
    return RetrievedContext(
        citation_id=citation_id,
        chunk_id=f"chunk-{citation_id}",
        document_id=f"doc-{citation_id}",
        filename=filename,
        position=0,
        text=text,
        retrieval_score=0.9,
        score=0.9,
    )


def test_retrieval_metrics_use_evidence_groups_and_ranking():
    trace = EvaluationTrace(
        question_id="q1",
        retrieved=[
            _chunk(1, "release.md", "Version 2 launched."),
            _chunk(2, "noise.md", "Unrelated."),
            _chunk(3, "approval.md", "Alex approved it."),
        ],
        answer="Version 2 launched and Alex approved it. [1] [3]",
        retrieval_ms=10,
        generation_ms=20,
    )

    metrics = evaluate_trace(_case(), trace)

    assert metrics["mrr"] == 1.0
    assert metrics["recall@1"] == 0.5
    assert metrics["recall@3"] == 1.0
    assert metrics["complete_evidence@3"] == 1.0
    assert metrics["precision@3"] == 2 / 3
    ndcg = metrics["ndcg@3"]
    assert isinstance(ndcg, float)
    assert ndcg < 1.0


def test_invalid_raw_citation_is_not_hidden_by_resolved_citations():
    trace = EvaluationTrace(
        question_id="q1",
        retrieved=[_chunk(1, "release.md", "Version 2 launched.")],
        answer="Version 2 launched. [1] Alex approved it. [99]",
        resolved_citation_ids=[1],
        retrieval_ms=1,
        generation_ms=2,
    )

    metrics = evaluate_trace(_case(), trace)

    assert metrics["citation_count"] == 2
    assert metrics["citation_validity"] == 0.5


def test_token_f1_normalizes_case_and_punctuation():
    assert token_f1("The answer is Blue.", "blue answer") == 2 * 2 / 6


def test_unanswerable_refusal_is_scored_without_retrieval_metrics():
    case = BenchmarkCase(
        id="q2",
        question="Who founded it?",
        answerable=False,
        expected_behavior="refuse",
    )
    trace = EvaluationTrace(
        question_id="q2",
        answer="I cannot answer because there is insufficient evidence.",
        retrieval_ms=2,
        generation_ms=3,
    )

    metrics = evaluate_trace(case, trace)

    assert "mrr" not in metrics
    assert metrics["refusal_correct"] == 1.0
    assert metrics["exact_match"] is None

from note_rag.evaluation.metrics import evaluate_trace, token_f1
from note_rag.evaluation.models import (
    BenchmarkCase,
    EvaluationTrace,
    RetrievedContext,
)


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

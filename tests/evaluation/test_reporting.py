import json

from note_rag.evaluation.models import (
    BenchmarkCase,
    EvaluationTrace,
    ExperimentSnapshot,
    QuestionResult,
)
from note_rag.evaluation.reporting import aggregate, write_reports


def _result(total_ms: float) -> QuestionResult:
    return QuestionResult(
        case=BenchmarkCase(
            id=f"q-{total_ms}",
            question="No answer?",
            answerable=False,
            expected_behavior="refuse",
        ),
        trace=EvaluationTrace(
            question_id=f"q-{total_ms}",
            answer="There is not enough information.",
            retrieval_ms=total_ms / 2,
            generation_ms=total_ms / 2,
        ),
        metrics={
            "total_ms": total_ms,
            "refusal_correct": 1.0,
            "estimated_cost_usd": None,
        },
    )


def test_aggregate_ignores_none_and_adds_latency_percentiles():
    report = aggregate([_result(10), _result(30)])

    assert report["metrics"]["total_ms"]["mean"] == 20
    assert report["metrics"]["total_ms"]["p50"] == 20
    assert report["metrics"]["total_ms"]["p95"] == 29
    assert "estimated_cost_usd" not in report["metrics"]


def test_write_reports_creates_comparison_artifacts(tmp_path):
    snapshot = ExperimentSnapshot(
        experiment="test",
        benchmark_path="benchmark.jsonl",
        benchmark_sha256="abc",
        benchmark_cases=1,
        corpus_manifest=None,
        corpus_sha256=None,
        base_url=None,
        git_commit="123",
        created_at="2026-01-01T00:00:00+00:00",
        retrieval={},
        ragas={"enabled": False},
    )

    write_reports(tmp_path, snapshot, [_result(10)])

    assert {
        "config.json",
        "metrics.json",
        "per_question.jsonl",
        "results.csv",
        "failures.jsonl",
    } <= {path.name for path in tmp_path.iterdir()}
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["questions"] == 1

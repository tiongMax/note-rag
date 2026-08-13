from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from note_rag.evaluation import aggregate_ragas_scores


def _load_runner() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "run_ragas_evaluation.py"
    spec = importlib.util.spec_from_file_location("ragas_runner", path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def test_aggregates_ragas_scores() -> None:
    summary = aggregate_ragas_scores(
        [
            {"faithfulness": 1.0, "context_recall": 0.5},
            {"faithfulness": 0.5, "context_recall": 1.0},
        ]
    )

    assert summary == {
        "sample_count": 2,
        "faithfulness": 0.75,
        "context_recall": 0.75,
    }


def test_rejects_inconsistent_ragas_metrics() -> None:
    with pytest.raises(ValueError, match="same metrics"):
        aggregate_ragas_scores([{"faithfulness": 1.0}, {"context_recall": 1.0}])


class FakeScorer:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[dict[str, object]] = []

    def score(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(value=self.value, reason="test reason")


def test_runner_maps_samples_to_modern_ragas_metric_inputs() -> None:
    runner = _load_runner()
    faithfulness = FakeScorer(0.8)
    precision = FakeScorer(0.6)

    results = runner.evaluate_samples(
        [
            {
                "id": "q1",
                "user_input": "question",
                "response": "answer",
                "reference": "reference",
                "retrieved_contexts": ["context"],
            }
        ],
        {"faithfulness": faithfulness, "context_precision": precision},
    )

    assert faithfulness.calls == [
        {
            "user_input": "question",
            "response": "answer",
            "retrieved_contexts": ["context"],
        }
    ]
    assert precision.calls == [
        {
            "user_input": "question",
            "reference": "reference",
            "retrieved_contexts": ["context"],
        }
    ]
    assert results[0]["ragas"] == {
        "faithfulness": 0.8,
        "context_precision": 0.6,
    }
    assert results[0]["ragas_reasons"] == {
        "faithfulness": "test reason",
        "context_precision": "test reason",
    }


class FakeClient:
    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert method == "POST"
        assert payload is not None
        assert payload["filters"] == {"filenames": ["source.pdf"]}
        if path == "/api/v1/retrieval/context":
            return {
                "chunks": [{"chunk_id": "chunk-1", "text": "support"}],
                "token_count": 1,
                "truncated": False,
            }
        assert path == "/api/v1/chat"
        return {
            "answer": "supported answer [1]",
            "model_name": "fake-chat",
            "conversation_id": "conversation-1",
            "citations": [{"chunk_id": "chunk-1"}],
        }


def test_collects_replayable_samples_and_latency_metrics() -> None:
    runner = _load_runner()
    entries = [
        {
            "id": "q1",
            "query": "question",
            "expected_answer": "answer",
            "relevant_passages": [{"source_id": "source.pdf"}],
        }
    ]

    samples = runner.collect_samples(
        client=FakeClient(),
        entries=entries,
        mode="hybrid",
        candidate_k=20,
        max_chunks=8,
        max_context_tokens=1200,
        vector_weight=0.7,
        rerank=True,
        rerank_weight=None,
    )
    runner.validate_samples(samples, entries)
    summary = runner.summarize_samples(samples)

    assert samples[0]["response"] == "supported answer [1]"
    assert samples[0]["retrieved_contexts"] == ["support"]
    assert summary["sample_count"] == 1
    assert summary["context_latency_count"] == 1
    assert summary["chat_latency_count"] == 1
    assert summary["context_tokens_mean"] == 1


def test_rejects_samples_from_a_different_dataset() -> None:
    runner = _load_runner()
    with pytest.raises(ValueError, match="IDs and order"):
        runner.validate_samples([{"id": "wrong"}], [{"id": "expected"}])

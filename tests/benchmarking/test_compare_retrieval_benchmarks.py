from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_script() -> ModuleType:
    path = (
        Path(__file__).parents[2]
        / "scripts"
        / "compare_retrieval_benchmarks.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_compare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(
    label: str,
    recall: float,
    *,
    ndcg: float = 0.5,
    irrelevant: float = 2.0,
) -> dict[str, Any]:
    return {
        "metadata": {
            "run_label": label,
            "dataset_sha256": "same",
            "mode": "hybrid",
            "top_k": 10,
            "vector_weight": 0.7,
            "relevance_threshold": 0.5,
            "repetitions": 1,
            "embedding_models": "same-model",
        },
        "metrics": {
            "query_count": 2,
            "recall_at_10": recall,
            "ndcg_at_10": ndcg,
            "irrelevant_at_5": irrelevant,
            "latency_p95_ms": 100.0,
        },
    }


def test_compares_controlled_runs() -> None:
    comparison_module = _load_script()

    comparison = comparison_module.compare(
        run("fixed", 0.5),
        run("recursive", 0.75),
    )

    recall = next(
        row
        for row in comparison["metrics"]
        if row["metric"] == "recall_at_10"
    )
    assert recall["absolute_delta"] == pytest.approx(0.25)
    assert recall["relative_change_percent"] == pytest.approx(50)
    assert recall["improved"] is True
    resume = comparison["resume_metrics"]
    assert resume["ndcg_at_10_relative_improvement_percent"] == 0
    assert resume["irrelevant_at_5_relative_reduction_percent"] == 0


def test_calculates_positive_resume_improvements() -> None:
    comparison_module = _load_script()

    comparison = comparison_module.compare(
        run("dense", 0.5, ndcg=0.5, irrelevant=2.0),
        run("candidate", 0.75, ndcg=0.6, irrelevant=1.5),
    )

    resume = comparison["resume_metrics"]
    assert resume["ndcg_at_10_relative_improvement_percent"] == pytest.approx(20)
    assert resume["irrelevant_at_5_relative_reduction_percent"] == pytest.approx(
        25
    )


def test_paired_bootstrap_reports_intervals_for_resume_metrics() -> None:
    comparison_module = _load_script()
    baseline = {
        "q1": {"metrics": {"ndcg_at_10": 0.5, "irrelevant_at_5": 4}},
        "q2": {"metrics": {"ndcg_at_10": 0.6, "irrelevant_at_5": 3}},
    }
    candidate = {
        "q1": {"metrics": {"ndcg_at_10": 0.6, "irrelevant_at_5": 3}},
        "q2": {"metrics": {"ndcg_at_10": 0.7, "irrelevant_at_5": 2}},
    }

    intervals = comparison_module.paired_bootstrap(
        baseline,
        candidate,
        samples=100,
        seed=7,
    )

    assert (
        intervals["ndcg_at_10_relative_improvement_percent"]["lower_95"] > 0
    )
    assert (
        intervals["irrelevant_at_5_relative_reduction_percent"]["lower_95"] > 0
    )


def test_rejects_different_experimental_controls() -> None:
    comparison_module = _load_script()
    candidate = run("recursive", 0.75)
    candidate["metadata"]["mode"] = "vector"

    with pytest.raises(ValueError, match="controls differ"):
        comparison_module.compare(run("fixed", 0.5), candidate)

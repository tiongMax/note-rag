from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_script() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "compare_retrieval_factorial.py"
    spec = importlib.util.spec_from_file_location("factorial_compare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(label: str, chunking: str, system: str, mrr: float) -> dict[str, Any]:
    return {
        "metadata": {
            "run_label": label,
            "dataset_sha256": "same",
            "mode": "hybrid",
            "system": system,
            "chunking_strategy": chunking,
            "lexical_backend": "bm25",
            "top_k": 20,
            "vector_weight": 0.7,
            "relevance_threshold": 0.5,
            "repetitions": 3,
            "embedding_models": "same-model",
        },
        "metrics": {
            "query_count": 48,
            "mrr": mrr,
            "recall_at_5": mrr,
            "ndcg_at_10": mrr,
        },
    }


def matrix() -> dict[str, dict[str, Any]]:
    return {
        "fixed_dense": run("a", "fixed", "vector", 0.5),
        "recursive_dense": run("b", "recursive", "vector", 0.6),
        "fixed_bm25_dense": run("c", "fixed", "api_bm25_dense", 0.7),
        "recursive_bm25_dense": run(
            "d", "recursive", "api_bm25_dense", 0.8
        ),
    }


def test_reports_isolated_and_combined_effects() -> None:
    script = _load_script()
    report = script.build_factorial_report(matrix())

    assert [effect["name"] for effect in report["effects"]] == [
        "chunking_at_dense",
        "bm25_at_fixed",
        "chunking_at_bm25",
        "combined",
    ]
    combined = report["effects"][-1]["metrics"]["mrr"]
    assert combined["baseline"] == 0.5
    assert combined["candidate"] == 0.8
    assert combined["relative_change_percent"] == pytest.approx(60)


def test_rejects_mislabeled_factorial_cell() -> None:
    script = _load_script()
    runs = matrix()
    runs["recursive_bm25_dense"]["metadata"]["chunking_strategy"] = "fixed"

    with pytest.raises(ValueError, match="chunking_strategy"):
        script.build_factorial_report(runs)


def test_paired_bootstrap_reports_combined_metric_intervals() -> None:
    script = _load_script()
    baseline = {
        "q1": {"mrr": 0.5, "recall_at_5": 0.5, "ndcg_at_10": 0.4},
        "q2": {"mrr": 0.6, "recall_at_5": 0.6, "ndcg_at_10": 0.5},
    }
    candidate = {
        "q1": {"mrr": 0.7, "recall_at_5": 0.8, "ndcg_at_10": 0.6},
        "q2": {"mrr": 0.8, "recall_at_5": 0.9, "ndcg_at_10": 0.7},
    }

    intervals = script.paired_bootstrap(
        baseline,
        candidate,
        samples=100,
        seed=7,
    )

    assert intervals["mrr"]["lower_95"] > 0
    assert intervals["recall_at_5"]["lower_95"] > 0
    assert intervals["ndcg_at_10"]["lower_95"] > 0

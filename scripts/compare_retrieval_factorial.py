"""Validate and summarize the fixed/recursive x dense/BM25 experiment."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_retrieval_benchmarks import _load, compare

_CELLS = {
    "fixed_dense": ("fixed", "vector"),
    "recursive_dense": ("recursive", "vector"),
    "fixed_bm25_dense": ("fixed", "api_bm25_dense"),
    "recursive_bm25_dense": ("recursive", "api_bm25_dense"),
}
_RESUME_METRICS = ("mrr", "recall_at_5", "ndcg_at_10")


def _query_metrics(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        query_id = value.get("id")
        metrics = value.get("metrics")
        if not isinstance(query_id, str) or not isinstance(metrics, dict):
            raise ValueError(f"{path}:{line_number}: invalid query result")
        rows[query_id] = {
            metric: float(metrics[metric]) for metric in _RESUME_METRICS
        }
    if not rows:
        raise ValueError(f"{path}: query results are empty")
    return rows


def paired_bootstrap(
    baseline: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
    *,
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, dict[str, float | int]]:
    """Estimate paired-query intervals for each combined relative effect."""

    if samples <= 0:
        raise ValueError("bootstrap samples must be greater than zero")
    if baseline.keys() != candidate.keys():
        raise ValueError("benchmark result query IDs differ")
    query_ids = sorted(baseline)
    randomizer = random.Random(seed)
    distributions = {metric: [] for metric in _RESUME_METRICS}
    for _ in range(samples):
        selected = [
            query_ids[randomizer.randrange(len(query_ids))]
            for _ in query_ids
        ]
        for metric in _RESUME_METRICS:
            baseline_mean = sum(
                baseline[query_id][metric] for query_id in selected
            ) / len(selected)
            candidate_mean = sum(
                candidate[query_id][metric] for query_id in selected
            ) / len(selected)
            distributions[metric].append(
                (candidate_mean - baseline_mean) / baseline_mean * 100
                if baseline_mean
                else 0.0
            )
    intervals: dict[str, dict[str, float | int]] = {}
    for metric, values in distributions.items():
        values.sort()
        intervals[metric] = {
            "lower_95": values[math.floor((len(values) - 1) * 0.025)],
            "upper_95": values[math.ceil((len(values) - 1) * 0.975)],
            "samples": samples,
            "seed": seed,
        }
    return intervals


def _effect(
    name: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    comparison = compare(baseline, candidate)
    rows = {row["metric"]: row for row in comparison["metrics"]}
    return {
        "name": name,
        "baseline": baseline["metadata"]["run_label"],
        "candidate": candidate["metadata"]["run_label"],
        "metrics": {
            metric: rows[metric]
            for metric in _RESUME_METRICS
            if metric in rows
        },
    }


def build_factorial_report(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Validate the four cells and report isolated plus combined effects."""

    if set(runs) != set(_CELLS):
        missing = sorted(set(_CELLS) - runs.keys())
        extra = sorted(runs.keys() - set(_CELLS))
        raise ValueError(f"factorial cells differ: missing={missing}, extra={extra}")
    for cell, (chunking, system) in _CELLS.items():
        metadata = runs[cell]["metadata"]
        if metadata.get("chunking_strategy") != chunking:
            raise ValueError(f"{cell}: chunking_strategy must be {chunking!r}")
        if metadata.get("system") != system:
            raise ValueError(f"{cell}: system must be {system!r}")
        if "bm25" in cell and metadata.get("lexical_backend") != "bm25":
            raise ValueError(f"{cell}: lexical_backend must be 'bm25'")

    baseline = runs["fixed_dense"]
    final = runs["recursive_bm25_dense"]
    effects = [
        _effect("chunking_at_dense", baseline, runs["recursive_dense"]),
        _effect("bm25_at_fixed", baseline, runs["fixed_bm25_dense"]),
        _effect("chunking_at_bm25", runs["fixed_bm25_dense"], final),
        _effect("combined", baseline, final),
    ]
    return {
        "schema_version": "1.0",
        "design": "2x2 chunking_strategy x retrieval_system",
        "cells": {
            name: {
                "run_label": run["metadata"]["run_label"],
                "chunking_strategy": run["metadata"]["chunking_strategy"],
                "system": run["metadata"]["system"],
                "lexical_backend": run["metadata"].get("lexical_backend"),
            }
            for name, run in runs.items()
        },
        "effects": effects,
    }


def _format(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        return str(value)
    return f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    for cell in _CELLS:
        parser.add_argument(f"--{cell.replace('_', '-')}", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixed-dense-results", type=Path)
    parser.add_argument("--recursive-bm25-dense-results", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    runs = {cell: _load(getattr(args, cell).resolve()) for cell in _CELLS}
    report = build_factorial_report(runs)
    if (args.fixed_dense_results is None) != (
        args.recursive_bm25_dense_results is None
    ):
        parser.error("both result paths are required for paired bootstrap")
    if args.fixed_dense_results is not None:
        report["combined_confidence_intervals"] = paired_bootstrap(
            _query_metrics(args.fixed_dense_results.resolve()),
            _query_metrics(args.recursive_bm25_dense_results.resolve()),
            samples=args.bootstrap_samples,
        )
    combined = next(
        effect for effect in report["effects"] if effect["name"] == "combined"
    )
    print("Combined fixed+dense -> recursive+BM25/dense:")
    for metric, row in combined["metrics"].items():
        print(
            f"- {metric}: {_format(row['baseline'])} -> {_format(row['candidate'])} "
            f"({_format(row['relative_change_percent'])}%)"
        )
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()

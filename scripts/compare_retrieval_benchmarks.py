"""Compare two controlled retrieval benchmark summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

_CONTROL_FIELDS = (
    "dataset_sha256",
    "mode",
    "top_k",
    "vector_weight",
    "relevance_threshold",
    "repetitions",
    "embedding_models",
)
_LOWER_IS_BETTER = {
    "embedding_budget_tokens",
    "irrelevant_at_5",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "rerank_latency_mean_ms",
    "rerank_latency_p50_ms",
    "rerank_latency_p95_ms",
    "retrieval_latency_mean_ms",
    "retrieval_latency_p50_ms",
    "retrieval_latency_p95_ms",
    "chunk_count",
}
_DESCRIPTIVE = {
    "bm25_build_ms",
    "chunk_tokens_max",
    "chunk_tokens_mean",
    "chunk_tokens_min",
    "chunk_tokens_p50",
    "chunk_tokens_p95",
}
_NON_COMPARATIVE = {
    "query_count",
    "request_count",
    "document_count",
    "document_token_count",
    "unstable_ranking_count",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value.get("metadata"), dict) or not isinstance(
        value.get("metrics"), dict
    ):
        raise ValueError(f"{path}: expected metadata and metrics objects")
    return value


def compare(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    mismatches = []
    for field in _CONTROL_FIELDS:
        baseline_value = baseline["metadata"].get(field)
        candidate_value = candidate["metadata"].get(field)
        if baseline_value != candidate_value:
            mismatches.append(
                f"{field}: {baseline_value!r} != {candidate_value!r}"
            )
    if mismatches:
        raise ValueError(
            "benchmark controls differ:\n"
            + "\n".join(f"- {mismatch}" for mismatch in mismatches)
        )

    rows = []
    shared_metrics = (
        baseline["metrics"].keys() & candidate["metrics"].keys()
    )
    for metric in sorted(shared_metrics - _NON_COMPARATIVE):
        baseline_value = baseline["metrics"][metric]
        candidate_value = candidate["metrics"][metric]
        if (
            isinstance(baseline_value, bool)
            or isinstance(candidate_value, bool)
            or not isinstance(baseline_value, (int, float))
            or not isinstance(candidate_value, (int, float))
        ):
            continue
        delta = candidate_value - baseline_value
        relative_change_percent = (
            delta / baseline_value * 100 if baseline_value else None
        )
        descriptive = metric in _DESCRIPTIVE
        lower_is_better = metric in _LOWER_IS_BETTER
        improved = (
            None
            if descriptive
            else (delta < 0 if lower_is_better else delta > 0)
        )
        rows.append(
            {
                "metric": metric,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "absolute_delta": delta,
                "relative_change_percent": relative_change_percent,
                "direction": (
                    "descriptive"
                    if descriptive
                    else (
                        "lower_is_better"
                        if lower_is_better
                        else "higher_is_better"
                    )
                ),
                "improved": improved,
            }
        )
    by_name = {row["metric"]: row for row in rows}
    ndcg_change = by_name.get("ndcg_at_10", {}).get(
        "relative_change_percent"
    )
    irrelevant_change = by_name.get("irrelevant_at_5", {}).get(
        "relative_change_percent"
    )
    return {
        "schema_version": "1.0",
        "baseline_label": baseline["metadata"]["run_label"],
        "candidate_label": candidate["metadata"]["run_label"],
        "controls": {
            field: baseline["metadata"].get(field) for field in _CONTROL_FIELDS
        },
        "metrics": rows,
        "resume_metrics": {
            "ndcg_at_10_relative_improvement_percent": ndcg_change,
            "irrelevant_at_5_relative_reduction_percent": (
                -irrelevant_change
                if irrelevant_change is not None
                else None
            ),
        },
    }


def _load_query_results(path: Path) -> dict[str, dict[str, Any]]:
    results = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        query_id = value.get("id")
        if not isinstance(query_id, str):
            raise ValueError(f"{path}:{line_number}: missing query id")
        results[query_id] = value
    return results


def paired_bootstrap(
    baseline_results: dict[str, dict[str, Any]],
    candidate_results: dict[str, dict[str, Any]],
    *,
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Return paired query-bootstrap intervals for the two résumé effects."""

    if samples <= 0:
        raise ValueError("bootstrap samples must be greater than zero")
    if baseline_results.keys() != candidate_results.keys():
        raise ValueError("benchmark result query IDs differ")
    query_ids = sorted(baseline_results)
    if not query_ids:
        raise ValueError("benchmark results are empty")
    randomizer = random.Random(seed)
    effects = {
        "ndcg_at_10_relative_improvement_percent": [],
        "irrelevant_at_5_relative_reduction_percent": [],
    }
    for _ in range(samples):
        selected = [
            query_ids[randomizer.randrange(len(query_ids))]
            for _ in query_ids
        ]
        baseline_ndcg = sum(
            float(baseline_results[query_id]["metrics"]["ndcg_at_10"])
            for query_id in selected
        ) / len(selected)
        candidate_ndcg = sum(
            float(candidate_results[query_id]["metrics"]["ndcg_at_10"])
            for query_id in selected
        ) / len(selected)
        baseline_irrelevant = sum(
            float(baseline_results[query_id]["metrics"]["irrelevant_at_5"])
            for query_id in selected
        ) / len(selected)
        candidate_irrelevant = sum(
            float(candidate_results[query_id]["metrics"]["irrelevant_at_5"])
            for query_id in selected
        ) / len(selected)
        effects["ndcg_at_10_relative_improvement_percent"].append(
            (candidate_ndcg - baseline_ndcg) / baseline_ndcg * 100
            if baseline_ndcg
            else 0.0
        )
        effects["irrelevant_at_5_relative_reduction_percent"].append(
            (baseline_irrelevant - candidate_irrelevant)
            / baseline_irrelevant
            * 100
            if baseline_irrelevant
            else 0.0
        )

    intervals = {}
    for metric, values in effects.items():
        values.sort()
        lower = values[math.floor((len(values) - 1) * 0.025)]
        upper = values[math.ceil((len(values) - 1) * 0.975)]
        intervals[metric] = {
            "lower_95": lower,
            "upper_95": upper,
            "samples": samples,
            "seed": seed,
        }
    return intervals


def _results_path(summary_path: Path) -> Path:
    name = summary_path.name
    if not name.endswith(".summary.json"):
        return summary_path.with_suffix(".results.jsonl")
    return summary_path.with_name(name.removesuffix(".summary.json") + ".results.jsonl")


def _format(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        return str(value)
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()

    comparison = compare(
        _load(args.baseline.resolve()),
        _load(args.candidate.resolve()),
    )
    baseline_results_path = _results_path(args.baseline.resolve())
    candidate_results_path = _results_path(args.candidate.resolve())
    if baseline_results_path.is_file() and candidate_results_path.is_file():
        comparison["confidence_intervals"] = paired_bootstrap(
            _load_query_results(baseline_results_path),
            _load_query_results(candidate_results_path),
        )
    print(
        f"{comparison['baseline_label']} -> {comparison['candidate_label']}"
    )
    for row in comparison["metrics"]:
        print(
            f"{row['metric']}: {_format(row['baseline'])} -> "
            f"{_format(row['candidate'])} "
            f"(delta {_format(row['absolute_delta'])}, "
            f"relative {_format(row['relative_change_percent'])}%)"
        )
    resume_metrics = comparison["resume_metrics"]
    print(
        "Résumé values: "
        f"X={_format(resume_metrics['ndcg_at_10_relative_improvement_percent'])}% "
        f"NDCG@10 improvement, "
        f"Y={_format(resume_metrics['irrelevant_at_5_relative_reduction_percent'])}% "
        "irrelevant@5 reduction"
    )
    for metric, interval in comparison.get("confidence_intervals", {}).items():
        print(
            f"{metric} 95% CI: "
            f"[{_format(interval['lower_95'])}%, "
            f"{_format(interval['upper_95'])}%]"
        )

    if args.output_prefix is None:
        return
    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    json_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    rows = comparison["metrics"]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()

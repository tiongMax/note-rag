"""JSON, JSONL, CSV, and aggregate experiment reports."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from note_rag.evaluation.models import (
    ExperimentSnapshot,
    QuestionResult,
)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def aggregate(results: list[QuestionResult]) -> dict[str, Any]:
    values: dict[str, list[float]] = defaultdict(list)
    errors = 0
    for result in results:
        errors += int(result.trace.error is not None)
        for prefix, scores in (
            ("", result.metrics),
            ("ragas.", result.judge),
        ):
            for name, value in scores.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if math.isfinite(float(value)):
                        values[f"{prefix}{name}"].append(float(value))
    metrics = {
        name: {
            "mean": _mean(items),
            "count": len(items),
        }
        for name, items in sorted(values.items())
    }
    for name in ("retrieval_ms", "generation_ms", "total_ms"):
        if name in values:
            metrics[name]["p50"] = _percentile(values[name], 0.5)
            metrics[name]["p95"] = _percentile(values[name], 0.95)
    return {
        "questions": len(results),
        "execution_errors": errors,
        "metrics": metrics,
    }


def _is_failure(result: QuestionResult) -> bool:
    if result.trace.error:
        return True
    checks = (
        result.metrics.get("recall@5"),
        result.metrics.get("citation_validity"),
        result.metrics.get("refusal_correct"),
    )
    if any(value is not None and value < 1.0 for value in checks):
        return True
    for name in ("faithfulness", "answer_correctness"):
        value = result.judge.get(name)
        if isinstance(value, (int, float)) and value < 0.8:
            return True
    return False


def write_reports(
    output_dir: Path,
    snapshot: ExperimentSnapshot,
    results: list[QuestionResult],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_report = aggregate(results)
    _write_json(output_dir / "config.json", snapshot.model_dump(mode="json"))
    _write_json(output_dir / "metrics.json", aggregate_report)
    _write_jsonl(output_dir / "per_question.jsonl", results)
    _write_jsonl(
        output_dir / "failures.jsonl",
        [result for result in results if _is_failure(result)],
    )
    _write_csv(output_dir / "results.csv", results)
    return aggregate_report


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, results: list[QuestionResult]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(result.model_dump_json() + "\n")


def _write_csv(path: Path, results: list[QuestionResult]) -> None:
    metric_names = sorted({name for result in results for name in result.metrics})
    judge_names = sorted(
        {
            name
            for result in results
            for name, value in result.judge.items()
            if not name.endswith(("_reason", "_error")) and not isinstance(value, str)
        }
    )
    fields = [
        "question_id",
        "answerable",
        "tags",
        "answer",
        "error",
        *metric_names,
        *(f"ragas.{name}" for name in judge_names),
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row: dict[str, Any] = {
                "question_id": result.case.id,
                "answerable": result.case.answerable,
                "tags": ",".join(result.case.tags),
                "answer": result.trace.answer,
                "error": result.trace.error,
                **result.metrics,
                **{f"ragas.{name}": result.judge.get(name) for name in judge_names},
            }
            writer.writerow(row)

"""CSV persistence for evaluation results."""

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from note_rag.evaluation.models import (
    GenerationResult,
    GenerationSummary,
    QuestionResult,
    SummaryResult,
)


def _detail_path(summary_path: Path, label: str) -> Path:
    suffix = summary_path.suffix or ".csv"
    return summary_path.with_name(f"{summary_path.stem}_{label}{suffix}")


def save_comparison_csv(path: Path, summaries: list[SummaryResult]) -> None:
    """Save aggregate ranked-retrieval metrics to CSV."""

    _write_csv(
        path,
        list(SummaryResult.__annotations__),
        cast("list[dict[str, Any]]", summaries),
    )


def save_retrieval_details_csv(
    path: Path, question_results: list[QuestionResult]
) -> Path:
    """Save auditable per-question retrieval results beside the summary."""

    detail_path = _detail_path(path, "details")
    rows: list[dict[str, Any]] = []
    for result in question_results:
        row = dict(result)
        row["relevant_chunk_ids"] = json.dumps(result["relevant_chunk_ids"])
        row["returned_chunk_ids"] = json.dumps(result["returned_chunk_ids"])
        row["relevant_ranks"] = json.dumps(result["relevant_ranks"])
        rows.append(row)
    _write_csv(detail_path, list(QuestionResult.__annotations__), rows)
    return detail_path


def save_generation_csvs(
    retrieval_summary_path: Path,
    results: list[GenerationResult],
    summaries: list[GenerationSummary],
) -> tuple[Path, Path]:
    """Save end-to-end detail and summary CSV files."""

    detail_path = _detail_path(retrieval_summary_path, "generation_details")
    summary_path = _detail_path(retrieval_summary_path, "generation_summary")
    detail_rows: list[dict[str, Any]] = []
    for result in results:
        row = dict(result)
        row["returned_chunk_ids"] = json.dumps(result["returned_chunk_ids"])
        detail_rows.append(row)
    _write_csv(detail_path, list(GenerationResult.__annotations__), detail_rows)
    _write_csv(
        summary_path,
        list(GenerationSummary.__annotations__),
        cast("list[dict[str, Any]]", summaries),
    )
    return detail_path, summary_path


def _write_csv(
    path: Path, fieldnames: list[str], rows: Sequence[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

"""Shared, deterministic artifacts for offline evaluation runners."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SAFE_LABEL = re.compile(r"^[A-Za-z0-9._-]+$")


def validate_run_label(label: str) -> str:
    """Validate a label before using it in an artifact filename."""

    if not _SAFE_LABEL.fullmatch(label):
        raise ValueError(
            "run label may contain only letters, digits, '.', '_', and '-'"
        )
    return label


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    """Load JSON objects from JSONL with useful line-numbered failures."""

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            message = f"{path}:{line_number}: invalid JSON: {error.msg}"
            raise ValueError(message) from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        records.append(value)
    if not records and not allow_empty:
        raise ValueError(f"{path}: dataset is empty")
    return records


def build_run_metadata(
    *,
    label: str,
    dataset_path: Path,
    configuration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the provenance block shared by every evaluation summary."""

    validate_run_label(label)
    return {
        "schema_version": "1.0",
        "run_label": label,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": dataset_path.resolve().as_posix(),
        "dataset_sha256": sha256_file(dataset_path),
        **dict(configuration or {}),
    }


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str)
            + "\n"
            for record in records
        ),
        encoding="utf-8",
        newline="\n",
    )
    return path


def write_evaluation_artifacts(
    *,
    output_dir: Path,
    label: str,
    kind: str,
    results: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Write per-sample JSONL and a compact summary JSON."""

    validate_run_label(label)
    validate_run_label(kind)
    results_path = output_dir / f"{label}.{kind}.results.jsonl"
    summary_path = output_dir / f"{label}.{kind}.summary.json"
    write_jsonl(results_path, results)
    summary_path.write_text(
        json.dumps(
            {"metadata": dict(metadata), "metrics": dict(metrics)},
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return results_path, summary_path

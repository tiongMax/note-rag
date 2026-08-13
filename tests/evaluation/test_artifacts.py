from __future__ import annotations

import json
from pathlib import Path

import pytest

from note_rag.evaluation import (
    build_run_metadata,
    load_jsonl,
    sha256_file,
    validate_run_label,
    write_evaluation_artifacts,
)


def test_loads_jsonl_and_reports_invalid_line(tmp_path: Path) -> None:
    valid = tmp_path / "valid.jsonl"
    valid.write_text('{"id":"one"}\n\n{"id":"two"}\n', encoding="utf-8")
    assert [row["id"] for row in load_jsonl(valid)] == ["one", "two"]

    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text('{"id":"one"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"invalid\.jsonl:2: invalid JSON"):
        load_jsonl(invalid)


def test_validates_artifact_labels() -> None:
    assert validate_run_label("recursive-v1.2") == "recursive-v1.2"
    with pytest.raises(ValueError, match="run label"):
        validate_run_label("../../escape")


def test_writes_provenance_and_evaluation_artifacts(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"id":"q1"}\n', encoding="utf-8", newline="\n")
    metadata = build_run_metadata(
        label="run-one",
        dataset_path=dataset,
        configuration={"mode": "hybrid"},
    )
    original_metadata = dict(metadata)

    results_path, summary_path = write_evaluation_artifacts(
        output_dir=tmp_path / "results",
        label="run-one",
        kind="quality",
        results=[{"id": "q1", "score": 1.0}],
        metadata=metadata,
        metrics={"sample_count": 1, "score": 1.0},
    )

    assert json.loads(results_path.read_text(encoding="utf-8"))["id"] == "q1"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["metadata"]["dataset_sha256"]
    assert summary["metadata"]["mode"] == "hybrid"
    assert summary["metadata"]["results_sha256"] == sha256_file(results_path)
    assert metadata == original_metadata
    assert "results_sha256" not in metadata
    assert summary["metrics"] == {"sample_count": 1, "score": 1.0}

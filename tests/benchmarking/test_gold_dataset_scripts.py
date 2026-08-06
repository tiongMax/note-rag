from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gold_dataset_is_nonempty_and_structurally_valid() -> None:
    repository_root = Path(__file__).parents[2]
    dataset_path = repository_root / "benchmarks/gold/dataset.jsonl"
    manifest_path = repository_root / "benchmarks/gold/manifest.json"

    records = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(records) >= 40
    assert len({record["id"] for record in records}) == len(records)
    assert manifest["question_count"] == len(records)
    assert all(record["schema_version"] == "1.0" for record in records)
    assert all(record["relevant_passages"] for record in records)
    assert all(
        passage["char_start"] < passage["char_end"]
        for record in records
        for passage in record["relevant_passages"]
    )


def test_sha256_accepts_text_and_bytes() -> None:
    builder = _load_script("build_gold_dataset")

    assert builder._sha256("same") == builder._sha256(b"same")

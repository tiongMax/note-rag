"""Validate the gold dataset schema, hashes, and source character offsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from note_rag.ingest import ParserRegistry  # noqa: E402


def _sha256(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        records.append(value)
    return records


def validate(dataset_path: Path, manifest_path: Path) -> tuple[int, int]:
    entries = _records(dataset_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    ids = [entry.get("id") for entry in entries]
    if len(ids) != len(set(ids)):
        errors.append("question IDs are not unique")
    if manifest.get("question_count") != len(entries):
        errors.append("manifest question_count does not match dataset")
    dataset_text = dataset_path.read_text(encoding="utf-8")
    if manifest.get("dataset_sha256") != _sha256(dataset_text):
        errors.append("manifest dataset_sha256 does not match dataset")

    parser_registry = ParserRegistry()
    parsed_by_path: dict[str, Any] = {}
    source_manifest = {
        source["source_path"]: source for source in manifest.get("sources", [])
    }
    passage_count = 0

    for entry in entries:
        entry_id = entry.get("id", "<unknown>")
        for field in (
            "schema_version",
            "id",
            "query",
            "expected_answer",
            "relevant_passages",
            "tags",
        ):
            if not entry.get(field):
                errors.append(f"{entry_id}: missing or empty {field}")
        for passage in entry.get("relevant_passages", []):
            passage_count += 1
            source_path_value = passage.get("source_path")
            if source_path_value not in source_manifest:
                errors.append(f"{entry_id}: source missing from manifest")
                continue
            source_path = REPOSITORY_ROOT / source_path_value
            if not source_path.is_file():
                errors.append(f"{entry_id}: source file not found: {source_path_value}")
                continue
            if source_path_value not in parsed_by_path:
                content = source_path.read_bytes()
                parsed_by_path[source_path_value] = parser_registry.get(
                    source_path.name
                ).parse(content)
                expected_source_hash = source_manifest[source_path_value][
                    "source_sha256"
                ]
                if _sha256(content) != expected_source_hash:
                    errors.append(f"{source_path_value}: source hash changed")
                if (
                    _sha256(parsed_by_path[source_path_value].text)
                    != source_manifest[source_path_value]["parsed_text_sha256"]
                ):
                    errors.append(f"{source_path_value}: parsed text hash changed")
            parsed = parsed_by_path[source_path_value]
            start = passage.get("char_start")
            end = passage.get("char_end")
            if not isinstance(start, int) or not isinstance(end, int):
                errors.append(f"{entry_id}: invalid character coordinates")
                continue
            reference_text = passage.get("reference_text", "")
            if len(reference_text.split()) < 8:
                errors.append(f"{entry_id}: gold passage is suspiciously short")
            if parsed.text[start:end] != reference_text:
                errors.append(f"{entry_id}: passage text does not match source offsets")
            if _sha256(reference_text) != passage.get("text_sha256"):
                errors.append(f"{entry_id}: passage text hash does not match")
            pages = [
                span.metadata.get("page_number")
                for span in parsed.source_spans
                if span.char_start < end and span.char_end > start
            ]
            if pages != [passage.get("page_number")]:
                errors.append(f"{entry_id}: page number does not match source span")

    if manifest.get("passage_count") != passage_count:
        errors.append("manifest passage_count does not match dataset")
    if errors:
        raise ValueError("\n".join(f"- {error}" for error in errors))
    return len(entries), passage_count


def main() -> None:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--dataset",
        type=Path,
        default=REPOSITORY_ROOT / "benchmarks/gold/dataset.jsonl",
    )
    argument_parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT / "benchmarks/gold/manifest.json",
    )
    args = argument_parser.parse_args()
    question_count, passage_count = validate(
        args.dataset.resolve(),
        args.manifest.resolve(),
    )
    print(f"Validated {question_count} questions and {passage_count} passages.")


if __name__ == "__main__":
    main()

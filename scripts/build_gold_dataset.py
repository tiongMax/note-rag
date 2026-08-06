"""Build stable gold-passage offsets from human-authored PDF annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from note_rag.ingest import PdfParser  # noqa: E402


def _sha256(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        records.append(record)
    return records


def _passage_for_annotation(
    parsed_text: str,
    source_spans: tuple[Any, ...],
    annotation: dict[str, Any],
) -> dict[str, Any]:
    page_number = annotation["page_number"]
    matching = [
        span
        for span in source_spans
        if span.metadata.get("page_number") == page_number
    ]
    if len(matching) != 1:
        raise ValueError(
            f"{annotation['id']}: page {page_number} has {len(matching)} text spans"
        )
    page_span = matching[0]
    page_text = parsed_text[page_span.char_start : page_span.char_end]
    start_marker = annotation["start_marker"]
    end_marker = annotation["end_marker"]
    start = page_text.find(start_marker)
    if start < 0:
        raise ValueError(
            f"{annotation['id']}: start marker not found on page {page_number}"
        )
    if end_marker is None:
        end = len(page_text)
    else:
        end = page_text.find(end_marker, start + len(start_marker))
        if end < 0:
            raise ValueError(
                f"{annotation['id']}: end marker not found on page {page_number}"
            )
    reference_text = page_text[start:end].strip()
    char_start = page_span.char_start + start
    char_end = char_start + len(reference_text)
    if parsed_text[char_start:char_end] != reference_text:
        raise AssertionError(f"{annotation['id']}: source coordinate mismatch")
    return {
        "source_id": Path(annotation["source_path"]).name,
        "source_path": annotation["source_path"],
        "page_number": page_number,
        "char_start": char_start,
        "char_end": char_end,
        "reference_text": reference_text,
        "text_sha256": _sha256(reference_text),
    }


def build(
    annotations_path: Path,
    dataset_path: Path,
    manifest_path: Path,
) -> tuple[int, int]:
    annotations = _load_jsonl(annotations_path)
    ids = [item.get("id") for item in annotations]
    duplicate_ids = sorted(
        item_id for item_id, count in Counter(ids).items() if count > 1
    )
    if duplicate_ids:
        raise ValueError(f"duplicate annotation IDs: {duplicate_ids}")

    parser = PdfParser()
    parsed_sources: dict[str, tuple[Any, bytes]] = {}
    source_question_counts: Counter[str] = Counter()
    entries: list[dict[str, Any]] = []

    for annotation in annotations:
        required = {
            "id",
            "query",
            "expected_answer",
            "source_path",
            "page_number",
            "start_marker",
            "end_marker",
            "tags",
        }
        missing = sorted(required - annotation.keys())
        if missing:
            raise ValueError(f"{annotation.get('id', '<unknown>')}: missing {missing}")

        source_path = (REPOSITORY_ROOT / annotation["source_path"]).resolve()
        source_key = source_path.as_posix()
        if source_key not in parsed_sources:
            content = source_path.read_bytes()
            parsed_sources[source_key] = (parser.parse(content), content)
        parsed, _ = parsed_sources[source_key]

        passages = [
            _passage_for_annotation(
                parsed.text,
                parsed.source_spans,
                annotation,
            )
        ]
        for additional in annotation.get("additional_passages", []):
            passages.append(
                _passage_for_annotation(
                    parsed.text,
                    parsed.source_spans,
                    {**annotation, **additional},
                )
            )

        source_question_counts[annotation["source_path"]] += 1
        entries.append(
            {
                "schema_version": "1.0",
                "id": annotation["id"],
                "query": annotation["query"],
                "expected_answer": annotation["expected_answer"],
                "relevant_passages": passages,
                "tags": annotation["tags"],
            }
        )

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_text = "".join(
        json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
        for entry in entries
    )
    dataset_path.write_text(dataset_text, encoding="utf-8", newline="\n")

    sources = []
    for source_key, (parsed, content) in sorted(parsed_sources.items()):
        source_path = Path(source_key)
        relative_path = source_path.relative_to(REPOSITORY_ROOT).as_posix()
        sources.append(
            {
                "source_id": source_path.name,
                "source_path": relative_path,
                "source_sha256": _sha256(content),
                "parsed_text_sha256": _sha256(parsed.text),
                "page_count": parsed.metadata["page_count"],
                "parsed_characters": len(parsed.text),
                "question_count": source_question_counts[relative_path],
            }
        )
    manifest = {
        "schema_version": "1.0",
        "dataset": dataset_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "dataset_sha256": _sha256(dataset_text),
        "question_count": len(entries),
        "passage_count": sum(len(entry["relevant_passages"]) for entry in entries),
        "sources": sources,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return len(entries), manifest["passage_count"]


def main() -> None:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--annotations",
        type=Path,
        default=REPOSITORY_ROOT / "benchmarks/gold/annotations.jsonl",
    )
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
    question_count, passage_count = build(
        args.annotations.resolve(),
        args.dataset.resolve(),
        args.manifest.resolve(),
    )
    print(f"Built {question_count} questions with {passage_count} gold passages.")


if __name__ == "__main__":
    main()

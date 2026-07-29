"""Evaluation dataset loading and validation."""

import json
from pathlib import Path
from typing import Any

from note_rag.evaluation.models import EVAL_CATEGORIES, EvalCase


def load_eval_set(path: Path) -> list[EvalCase]:
    """Load and validate a JSON evaluation dataset."""

    with path.open(encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, list) or not raw:
        raise ValueError("eval JSON must be a non-empty list")
    return [_validate_case(value, index) for index, value in enumerate(raw)]


def _validate_case(value: Any, index: int) -> EvalCase:
    required = {"question", "answer", "relevant_chunk_ids", "category"}
    if not isinstance(value, dict) or not required.issubset(value):
        raise ValueError(f"eval item {index} is missing required fields")
    if value["category"] not in EVAL_CATEGORIES:
        raise ValueError(f"eval item {index} has an invalid category")
    if not isinstance(value["question"], str) or not isinstance(value["answer"], str):
        raise ValueError(f"eval item {index} question/answer must be strings")
    relevant_ids = value["relevant_chunk_ids"]
    if not isinstance(relevant_ids, list) or not all(
        isinstance(chunk_id, (str, int)) for chunk_id in relevant_ids
    ):
        raise ValueError(f"eval item {index} relevant_chunk_ids must be a list of IDs")
    category = str(value["category"])
    if category == "out-of-corpus" and relevant_ids:
        raise ValueError(f"eval item {index} is out-of-corpus but has relevant IDs")
    if category != "out-of-corpus" and not relevant_ids:
        raise ValueError(f"eval item {index} is in-corpus but has no relevant IDs")
    case: EvalCase = {
        "question": value["question"],
        "answer": value["answer"],
        "relevant_chunk_ids": [str(chunk_id) for chunk_id in relevant_ids],
        "category": category,
    }
    if "id" in value:
        case["id"] = str(value["id"])
    return case

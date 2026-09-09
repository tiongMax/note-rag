"""Strict JSONL benchmark loading and validation."""

import json
from pathlib import Path

from pydantic import ValidationError

from note_rag.evaluation.models import BenchmarkCase, EvaluationTrace


def load_benchmark(path: Path) -> list[BenchmarkCase]:
    """Load a non-empty benchmark and reject duplicate IDs."""

    cases: list[BenchmarkCase] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                case = BenchmarkCase.model_validate_json(line)
            except (ValidationError, ValueError) as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid benchmark row: {error}"
                ) from error
            if case.id in seen:
                raise ValueError(
                    f"{path}:{line_number}: duplicate case id {case.id!r}"
                )
            seen.add(case.id)
            cases.append(case)
    if not cases:
        raise ValueError(f"{path}: benchmark contains no cases")
    return cases


def load_traces(path: Path) -> dict[str, EvaluationTrace]:
    """Load previously captured traces keyed by question ID."""

    traces: dict[str, EvaluationTrace] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
                if "trace" in payload:
                    payload = payload["trace"]
                trace = EvaluationTrace.model_validate(payload)
            except (json.JSONDecodeError, ValidationError) as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid trace row: {error}"
                ) from error
            if trace.question_id in traces:
                raise ValueError(
                    f"{path}:{line_number}: duplicate trace for "
                    f"{trace.question_id!r}"
                )
            traces[trace.question_id] = trace
    return traces

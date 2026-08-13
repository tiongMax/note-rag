"""Evaluate deterministic guardrail rules and benchmark their local compute path.

The classification metrics produced here describe the repository's versioned
deterministic rules and lexical overlap proxies.  They are not semantic safety,
faithfulness, or answer-quality judgments.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from note_rag.chunking import RegexTokenCounter  # noqa: E402
from note_rag.context import ContextChunk, ContextPackage  # noqa: E402
from note_rag.evaluation import (  # noqa: E402
    load_jsonl,
    sha256_file,
    validate_run_label,
)
from note_rag.retrieval import SearchMode  # noqa: E402

STAGES = ("input", "context", "output")
DEFAULT_ITERATIONS = 10_000
DEFAULT_WARMUP_ITERATIONS = 1_000
_CASE_NAMESPACE = uuid.UUID("f350e3d6-a27f-5dc8-b15f-4a3bc96d53ea")


class GuardrailServiceLike(Protocol):
    def inspect_input(self, question: str) -> Any: ...

    def sanitize_context(self, context: ContextPackage) -> Any: ...

    def validate_output(
        self,
        *,
        question: str,
        answer: str,
        context: ContextPackage,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class GuardrailCase:
    schema_version: str
    ruleset_version: str
    id: str
    stage: str
    label: str
    category: str
    expected_unsafe: bool
    hard_negative: bool
    question: str
    contexts: tuple[str, ...]
    context_roles: tuple[str, ...]
    answer: str
    rationale: str
    tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkFixture:
    name: str
    question: str
    context: ContextPackage
    answer: str
    expected_question_tokens: int
    expected_context_tokens: int
    expected_answer_tokens: int


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_cases(path: Path) -> list[GuardrailCase]:
    """Load and strictly validate the synthetic regression case corpus."""

    records = load_jsonl(path)
    cases: list[GuardrailCase] = []
    ids: set[str] = set()
    for row_number, record in enumerate(records, start=1):
        prefix = f"{path}:{row_number}"
        required = {
            "schema_version",
            "ruleset_version",
            "id",
            "stage",
            "label",
            "category",
            "expected_unsafe",
            "hard_negative",
            "question",
            "contexts",
            "context_roles",
            "answer",
            "rationale",
            "tags",
        }
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(f"{prefix}: missing fields: {', '.join(missing)}")
        extra = sorted(record.keys() - required)
        if extra:
            raise ValueError(f"{prefix}: unexpected fields: {', '.join(extra)}")

        identifier = record["id"]
        stage = record["stage"]
        label = record["label"]
        category = record["category"]
        contexts = record["contexts"]
        context_roles = record["context_roles"]
        tags = record["tags"]
        for field in (
            "schema_version",
            "ruleset_version",
            "id",
            "stage",
            "label",
            "category",
            "question",
            "answer",
            "rationale",
        ):
            if not isinstance(record[field], str):
                raise ValueError(f"{prefix}: {field} must be a string")
        if not identifier or identifier in ids:
            raise ValueError(f"{prefix}: case ID must be non-empty and unique")
        if stage not in STAGES:
            raise ValueError(f"{prefix}: stage must be input, context, or output")
        if label not in {"safe", "unsafe"}:
            raise ValueError(f"{prefix}: label must be safe or unsafe")
        if not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", category):
            raise ValueError(f"{prefix}: category must be snake_case")
        if type(record["expected_unsafe"]) is not bool:
            raise ValueError(f"{prefix}: expected_unsafe must be boolean")
        if (label == "unsafe") is not record["expected_unsafe"]:
            raise ValueError(f"{prefix}: label must agree with expected_unsafe")
        if type(record["hard_negative"]) is not bool:
            raise ValueError(f"{prefix}: hard_negative must be boolean")
        if record["hard_negative"] and record["expected_unsafe"]:
            raise ValueError(f"{prefix}: a hard negative must be expected safe")
        if (
            not isinstance(contexts, list)
            or not contexts
            or not all(isinstance(item, str) and item for item in contexts)
        ):
            raise ValueError(f"{prefix}: contexts must be non-empty strings")
        if (
            not isinstance(context_roles, list)
            or len(context_roles) != len(contexts)
            or not all(
                isinstance(role, str)
                and role in {"supporting", "distractor", "adversarial"}
                for role in context_roles
            )
        ):
            raise ValueError(f"{prefix}: context_roles must align with contexts")
        if (
            not isinstance(tags, list)
            or not tags
            or not all(isinstance(item, str) and item for item in tags)
        ):
            raise ValueError(f"{prefix}: tags must be non-empty strings")
        if stage == "input" and not record["question"]:
            raise ValueError(f"{prefix}: input cases require a question")
        if stage == "context" and not contexts:
            raise ValueError(f"{prefix}: context cases require contexts")
        if stage == "output" and not record["answer"]:
            raise ValueError(f"{prefix}: output cases require an answer")

        ids.add(identifier)
        cases.append(
            GuardrailCase(
                schema_version=record["schema_version"],
                ruleset_version=record["ruleset_version"],
                id=identifier,
                stage=stage,
                label=label,
                category=category,
                expected_unsafe=record["expected_unsafe"],
                hard_negative=record["hard_negative"],
                question=record["question"],
                contexts=tuple(contexts),
                context_roles=tuple(context_roles),
                answer=record["answer"],
                rationale=record["rationale"],
                tags=tuple(tags),
            )
        )

    schema_versions = {case.schema_version for case in cases}
    ruleset_versions = {case.ruleset_version for case in cases}
    if len(schema_versions) != 1:
        raise ValueError("case corpus must use exactly one schema version")
    if len(ruleset_versions) != 1:
        raise ValueError("case corpus must target exactly one ruleset version")
    for stage in STAGES:
        labels = {case.expected_unsafe for case in cases if case.stage == stage}
        if labels != {False, True}:
            raise ValueError(f"stage {stage!r} must contain safe and unsafe cases")
    return cases


def validate_manifest(
    path: Path,
    *,
    cases_path: Path,
    cases: Sequence[GuardrailCase],
) -> dict[str, Any]:
    """Fail closed when the manifest no longer describes the case corpus."""

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error.msg}") from error
    if not isinstance(manifest, dict):
        raise ValueError("guardrail manifest must be a JSON object")
    if manifest.get("cases_sha256") != sha256_file(cases_path):
        raise ValueError("guardrail case corpus hash does not match manifest")
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("guardrail manifest counts must be a JSON object")
    if counts.get("overall") != len(cases):
        raise ValueError("guardrail case count does not match manifest")
    declaration = manifest.get("evaluation_declaration")
    if (
        manifest.get("synthetic") is not True
        or manifest.get("split") != "regression"
        or not isinstance(declaration, dict)
        or declaration.get("held_out") is not False
    ):
        raise ValueError(
            "guardrail manifest must declare synthetic regression data"
        )
    schema_versions = sorted({case.schema_version for case in cases})
    ruleset_versions = sorted({case.ruleset_version for case in cases})
    if manifest.get("schema_version") != schema_versions[0]:
        raise ValueError("case schema version does not match manifest")
    if manifest.get("ruleset_version") != ruleset_versions[0]:
        raise ValueError("ruleset version does not match manifest")

    stage_counts = dict(sorted(Counter(case.stage for case in cases).items()))
    label_counts = dict(sorted(Counter(case.label for case in cases).items()))
    unsafe_counts = {
        "safe": sum(not case.expected_unsafe for case in cases),
        "unsafe": sum(case.expected_unsafe for case in cases),
    }
    stage_label_counts = {
        stage: {
            label: sum(case.stage == stage and case.label == label for case in cases)
            for label in ("safe", "unsafe")
        }
        for stage in STAGES
    }
    if counts.get("by_stage") != stage_counts:
        raise ValueError("stage counts do not match manifest")
    if counts.get("by_label") != label_counts:
        raise ValueError("label counts do not match manifest")
    if counts.get("by_stage_and_label") != stage_label_counts:
        raise ValueError("stage and label counts do not match manifest")
    if label_counts != unsafe_counts:
        raise ValueError("case labels do not match expected outcomes")
    if counts.get("hard_negatives") != sum(case.hard_negative for case in cases):
        raise ValueError("hard-negative count does not match manifest")
    allowed_values = manifest.get("allowed_values")
    if not isinstance(allowed_values, dict) or (
        allowed_values.get("stage") != list(STAGES)
        or allowed_values.get("label") != ["safe", "unsafe"]
        or allowed_values.get("context_role")
        != ["supporting", "distractor", "adversarial"]
    ):
        raise ValueError("allowed values do not match the case schema")
    declared_cases = manifest.get("cases")
    if (
        not isinstance(declared_cases, str)
        or Path(declared_cases).name != cases_path.name
    ):
        raise ValueError("manifest cases path does not identify the case corpus")
    dataset_version = manifest.get("dataset_version")
    if not isinstance(dataset_version, str) or not dataset_version:
        raise ValueError("manifest dataset version must be non-empty")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("manifest dataset name must be non-empty")
    return manifest


def _token_count(text: str) -> int:
    return RegexTokenCounter().count(text)


def build_context(
    texts: Sequence[str],
    *,
    identity: str,
) -> ContextPackage:
    """Construct a deterministic synthetic context package for one case."""

    chunks: list[ContextChunk] = []
    rendered: list[str] = []
    for position, text in enumerate(texts):
        citation_id = position + 1
        chunk_id = uuid.uuid5(_CASE_NAMESPACE, f"{identity}:chunk:{position}")
        document_id = uuid.uuid5(_CASE_NAMESPACE, f"{identity}:document:{position}")
        chunks.append(
            ContextChunk(
                citation_id=citation_id,
                chunk_id=chunk_id,
                document_id=document_id,
                filename=f"synthetic-guardrail-{position + 1}.txt",
                media_type="text/plain",
                position=position,
                text=text,
                token_count=_token_count(text),
                source_metadata={"synthetic": True, "case_id": identity},
                retrieval_score=1.0,
                rerank_score=None,
                score=1.0,
            )
        )
        rendered.append(
            f"[{citation_id}] Source: synthetic-guardrail-{position + 1}.txt\n{text}"
        )
    context_text = "\n\n".join(rendered)
    return ContextPackage(
        query=identity,
        mode=SearchMode.HYBRID,
        context=context_text,
        chunks=chunks,
        token_count=_token_count(context_text),
        token_budget=max(1, _token_count(context_text)),
        candidates_considered=len(chunks),
        duplicates_removed=0,
        truncated=False,
        reranker_model=None,
    )


def _decision_action(decision: Any) -> str:
    action = getattr(decision, "action", None)
    value = getattr(action, "value", action)
    if not isinstance(value, str):
        raise TypeError("guardrail decision action must be a string-like enum")
    normalized = value.strip().lower()
    if "." in normalized:
        normalized = normalized.rsplit(".", 1)[-1]
    if normalized not in {"allow", "block", "filter"}:
        raise ValueError(f"unsupported guardrail action: {value!r}")
    return normalized


def _decision_stage(decision: Any) -> str:
    stage = getattr(decision, "stage", None)
    value = getattr(stage, "value", stage)
    if not isinstance(value, str):
        raise TypeError("guardrail decision stage must be a string-like enum")
    normalized = str(value).strip().lower()
    if "." in normalized:
        normalized = normalized.rsplit(".", 1)[-1]
    if normalized not in STAGES:
        raise ValueError(f"unsupported guardrail stage: {value!r}")
    return normalized


def _decision_reason_codes(decision: Any) -> list[str]:
    codes = getattr(decision, "reason_codes", ())
    if not isinstance(codes, (list, tuple)):
        raise TypeError("guardrail decision reason_codes must be a sequence")
    return [str(code) for code in codes]


def evaluate_case(
    service: GuardrailServiceLike,
    case: GuardrailCase,
) -> dict[str, Any]:
    """Evaluate one versioned regression case against its target stage."""

    context = build_context(case.contexts, identity=case.id)
    started = time.perf_counter_ns()
    details: dict[str, Any] = {}
    if case.stage == "input":
        decision = service.inspect_input(case.question)
    elif case.stage == "context":
        guarded = service.sanitize_context(context)
        decision = getattr(guarded, "decision")
        details["filtered_chunk_count"] = int(
            getattr(guarded, "filtered_chunk_count", 0)
        )
    else:
        guarded = service.validate_output(
            question=case.question,
            answer=case.answer,
            context=context,
        )
        decision = getattr(guarded, "decision")
        for field in (
            "lexical_groundedness_proxy",
            "lexical_relevance_proxy",
        ):
            value = getattr(guarded, field, None)
            details[field] = None if value is None else float(value)
    elapsed_ns = time.perf_counter_ns() - started
    action = _decision_action(decision)
    decision_stage = _decision_stage(decision)
    if decision_stage != case.stage:
        raise ValueError(
            f"case {case.id!r}: guardrail returned stage {decision_stage!r}, "
            f"expected {case.stage!r}"
        )
    predicted_unsafe = action != "allow"
    return {
        "id": case.id,
        "stage": case.stage,
        "label": case.label,
        "category": case.category,
        "expected_unsafe": case.expected_unsafe,
        "predicted_unsafe": predicted_unsafe,
        "correct": predicted_unsafe == case.expected_unsafe,
        "hard_negative": case.hard_negative,
        "action": action,
        "decision_stage": decision_stage,
        "reason_codes": _decision_reason_codes(decision),
        "measured_latency_ns": elapsed_ns,
        "reported_latency_ms": float(getattr(decision, "latency_ms", 0.0)),
        **details,
    }


def evaluate_cases(
    service: GuardrailServiceLike,
    cases: Sequence[GuardrailCase],
) -> list[dict[str, Any]]:
    return [evaluate_case(service, case) for case in cases]


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def _confusion(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tp = sum(
        row["expected_unsafe"] is True and row["predicted_unsafe"] is True
        for row in rows
    )
    fp = sum(
        row["expected_unsafe"] is False and row["predicted_unsafe"] is True
        for row in rows
    )
    tn = sum(
        row["expected_unsafe"] is False and row["predicted_unsafe"] is False
        for row in rows
    )
    fn = sum(
        row["expected_unsafe"] is True and row["predicted_unsafe"] is False
        for row in rows
    )
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {
        "case_count": len(rows),
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
        },
        "accuracy": _ratio(tp + tn, len(rows)),
        "precision": precision,
        "recall": recall,
        "specificity": _ratio(tn, tn + fp),
        "f1": _ratio(2 * precision * recall, precision + recall),
        "false_positive_ids": sorted(
            str(row["id"])
            for row in rows
            if row["expected_unsafe"] is False and row["predicted_unsafe"] is True
        ),
        "false_negative_ids": sorted(
            str(row["id"])
            for row in rows
            if row["expected_unsafe"] is True and row["predicted_unsafe"] is False
        ),
    }


def summarize_classification(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one classification result is required")
    return {
        "metric_scope": (
            "deterministic rule and lexical proxy classification; not a "
            "semantic safety evaluation"
        ),
        "overall": _confusion(rows),
        "by_stage": {
            stage: _confusion([row for row in rows if row["stage"] == stage])
            for stage in STAGES
        },
        "hard_negative_false_positive_ids": sorted(
            str(row["id"])
            for row in rows
            if row["hard_negative"] and row["predicted_unsafe"]
        ),
    }


def _repeated_tokens(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{index}" for index in range(count))


def build_benchmark_fixtures() -> dict[str, BenchmarkFixture]:
    """Return normal and declared max-size complete-path fixtures."""

    normal_question = "Summarize the attendance policy for a new student."
    normal_contexts = (
        "The attendance policy allows two excused absences per term.",
        "Students should notify the office before an excused absence.",
    )
    normal_answer = (
        "The policy allows two excused absences, with advance notice to the "
        "office [1] [2]."
    )

    max_question = _repeated_tokens("question", 512)
    max_contexts = tuple(
        _repeated_tokens(f"context{chunk}_", 150) for chunk in range(8)
    )
    max_answer = _repeated_tokens("answer", 1024)
    return {
        "normal": BenchmarkFixture(
            name="normal",
            question=normal_question,
            context=build_context(normal_contexts, identity="fixture-normal"),
            answer=normal_answer,
            expected_question_tokens=_token_count(normal_question),
            expected_context_tokens=sum(_token_count(text) for text in normal_contexts),
            expected_answer_tokens=_token_count(normal_answer),
        ),
        "max": BenchmarkFixture(
            name="max",
            question=max_question,
            context=build_context(max_contexts, identity="fixture-max"),
            answer=max_answer,
            expected_question_tokens=512,
            expected_context_tokens=1200,
            expected_answer_tokens=1024,
        ),
    }


def run_complete_guardrail_path(
    service: GuardrailServiceLike,
    fixture: BenchmarkFixture,
) -> tuple[str, str, str]:
    """Run input, context, and output stages exactly once, entirely locally."""

    input_decision = service.inspect_input(fixture.question)
    context_result = service.sanitize_context(fixture.context)
    sanitized = getattr(context_result, "context")
    output_result = service.validate_output(
        question=fixture.question,
        answer=fixture.answer,
        context=sanitized,
    )
    return (
        _decision_action(input_decision),
        _decision_action(getattr(context_result, "decision")),
        _decision_action(getattr(output_result, "decision")),
    )


def _percentile(values: Sequence[int], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("latency distribution cannot be empty")
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_latency_ns(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        raise ValueError("at least one latency measurement is required")
    median_ns = float(statistics.median(values))
    p95_ns = _percentile(values, 0.95)
    p99_ns = _percentile(values, 0.99)
    return {
        "count": len(values),
        "median_ns": median_ns,
        "p95_ns": p95_ns,
        "p99_ns": p99_ns,
        "median_ms": median_ns / 1_000_000,
        "p95_ms": p95_ns / 1_000_000,
        "p99_ms": p99_ns / 1_000_000,
        "min_ns": min(values),
        "max_ns": max(values),
    }


def benchmark_fixture(
    service: GuardrailServiceLike,
    fixture: BenchmarkFixture,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
) -> dict[str, Any]:
    """Benchmark the complete three-stage path using ``perf_counter_ns``."""

    if iterations <= 0:
        raise ValueError("iterations must be greater than zero")
    if warmup_iterations < 0:
        raise ValueError("warmup iterations cannot be negative")
    for _ in range(warmup_iterations):
        run_complete_guardrail_path(service, fixture)

    latencies: list[int] = []
    final_actions: tuple[str, str, str] | None = None
    for _ in range(iterations):
        started = time.perf_counter_ns()
        final_actions = run_complete_guardrail_path(service, fixture)
        latencies.append(time.perf_counter_ns() - started)
    assert final_actions is not None
    return {
        "fixture": fixture.name,
        "iterations": iterations,
        "warmup_iterations": warmup_iterations,
        "clock": "time.perf_counter_ns",
        "complete_path_stages": list(STAGES),
        "question_tokens": fixture.expected_question_tokens,
        "context_body_tokens": fixture.expected_context_tokens,
        "context_chunks": len(fixture.context.chunks),
        "answer_tokens": fixture.expected_answer_tokens,
        "final_actions": list(final_actions),
        "latency": summarize_latency_ns(latencies),
    }


def machine_provenance() -> dict[str, Any]:
    machine = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    python = {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "compiler": platform.python_compiler(),
    }
    return {
        "machine": machine,
        "machine_sha256": _canonical_sha256(machine),
        "python": python,
        "python_sha256": _canonical_sha256(python),
        "perf_counter_resolution_seconds": time.get_clock_info(
            "perf_counter"
        ).resolution,
    }


def repository_provenance(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Bind a report to exact local source files and Git state when available."""

    source_paths = (
        repository_root / "src/note_rag/guardrails/models.py",
        repository_root / "src/note_rag/guardrails/rules.py",
        repository_root / "src/note_rag/guardrails/service.py",
        repository_root / "src/note_rag/chat/service.py",
    )
    source_hashes = {
        path.relative_to(repository_root).as_posix(): sha256_file(path)
        for path in source_paths
    }

    def git_output(*arguments: str) -> str | None:
        command = [
            "git",
            "-c",
            f"safe.directory={repository_root.resolve().as_posix()}",
            *arguments,
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    commit = git_output("rev-parse", "HEAD")
    status = git_output("status", "--porcelain", "--untracked-files=normal")
    return {
        "git_commit": commit or None,
        "git_dirty": None if status is None else bool(status),
        "source_sha256": source_hashes,
        "combined_source_sha256": _canonical_sha256(source_hashes),
    }


def build_report(
    *,
    label: str,
    cases_path: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    ruleset_version: str,
    rows: Sequence[Mapping[str, Any]],
    benchmarks: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    validate_run_label(label)
    corpus_hashes = {
        "cases_sha256": sha256_file(cases_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    corpus_hashes["combined_sha256"] = _canonical_sha256(corpus_hashes)
    return {
        "schema_version": "1.0",
        "run_label": label,
        "ruleset_version": ruleset_version,
        "dataset_kind": "versioned synthetic guardrail regression cases",
        "metric_scope": (
            "deterministic rules and lexical overlap proxies only; results "
            "must not be described as semantic safety or semantic grounding"
        ),
        "manifest": dict(manifest),
        "corpus_hashes": corpus_hashes,
        "runtime": machine_provenance(),
        "implementation": repository_provenance(),
        "classification": summarize_classification(rows),
        "latency_benchmarks": dict(benchmarks),
    }


def write_report(
    output_dir: Path,
    *,
    label: str,
    rows: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{label}.guardrail.results.jsonl"
    summary_path = output_dir / f"{label}.guardrail.summary.json"
    results_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )
    payload = {
        **dict(report),
        "results_sha256": sha256_file(results_path),
    }
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return results_path, summary_path


def create_guardrail_service() -> tuple[GuardrailServiceLike, str]:
    """Load the production guardrail service lazily for standalone execution."""

    try:
        module = importlib.import_module("note_rag.guardrails")
        service_class = getattr(module, "GuardrailService")
        ruleset_version = getattr(module, "GUARDRAIL_RULESET_VERSION")
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "note_rag.guardrails.GuardrailService and "
            "GUARDRAIL_RULESET_VERSION are required"
        ) from error
    return service_class(), str(ruleset_version)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="guardrails")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPOSITORY_ROOT / "benchmarks/guardrails/cases.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT / "benchmarks/guardrails/manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "tmp/benchmarks",
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=DEFAULT_WARMUP_ITERATIONS,
    )
    parser.add_argument(
        "--fixture",
        choices=("all", "normal", "max"),
        default="all",
    )
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        validate_run_label(args.label)
    except ValueError as error:
        parser.error(str(error))
    if args.iterations <= 0:
        parser.error("--iterations must be greater than zero")
    if args.warmup_iterations < 0:
        parser.error("--warmup-iterations cannot be negative")

    cases_path = args.dataset.resolve()
    manifest_path = args.manifest.resolve()
    cases = load_cases(cases_path)
    manifest = validate_manifest(
        manifest_path,
        cases_path=cases_path,
        cases=cases,
    )
    service, ruleset_version = create_guardrail_service()
    if ruleset_version != manifest["ruleset_version"]:
        parser.error(
            "service ruleset version does not match the regression corpus: "
            f"{ruleset_version!r} != {manifest['ruleset_version']!r}"
        )

    rows = evaluate_cases(service, cases)
    fixtures = build_benchmark_fixtures()
    selected = (
        fixtures if args.fixture == "all" else {args.fixture: fixtures[args.fixture]}
    )
    benchmarks = {
        name: benchmark_fixture(
            service,
            fixture,
            iterations=args.iterations,
            warmup_iterations=args.warmup_iterations,
        )
        for name, fixture in selected.items()
    }
    report = build_report(
        label=args.label,
        cases_path=cases_path,
        manifest_path=manifest_path,
        manifest=manifest,
        ruleset_version=ruleset_version,
        rows=rows,
        benchmarks=benchmarks,
    )
    paths = write_report(
        args.output_dir.resolve(),
        label=args.label,
        rows=rows,
        report=report,
    )
    print(json.dumps(report["classification"], indent=2))
    print("Wrote:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()

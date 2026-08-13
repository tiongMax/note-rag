"""Collect grounded answers and evaluate them with RAGAS."""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from run_retrieval_benchmark import ApiClient  # noqa: E402

from note_rag.evaluation import (  # noqa: E402
    aggregate_numeric_scores,
    aggregate_ragas_scores,
    build_run_metadata,
    evaluate_deterministic_answer_quality,
    load_jsonl,
    sha256_file,
    summarize_latencies,
    validate_run_label,
    write_evaluation_artifacts,
)

_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "context_precision",
    "context_recall",
)
_PRODUCER_CONTROL_FIELDS = (
    "mode",
    "candidate_k",
    "max_chunks",
    "max_context_tokens",
    "vector_weight",
    "rerank",
    "rerank_weight",
    "expected_chunking_strategy",
    "expected_lexical_backend",
    "expected_answer_model",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class RagasScorer(Protocol):
    def score(self, **kwargs: Any) -> Any: ...


def _safe_metadata_path(
    path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> str:
    """Render repository paths without leaking an absolute workspace path."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return resolved.name


def runner_git_provenance(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, str | bool | None]:
    """Return this runner's commit provenance without failing outside Git."""

    root = repository_root.resolve()

    def output(arguments: Sequence[str]) -> str | None:
        command = [
            "git",
            "-c",
            f"safe.directory={root.as_posix()}",
            *arguments,
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    commit = output(("rev-parse", "HEAD"))
    status = output(("status", "--porcelain", "--untracked-files=normal"))
    return {
        "runner_git_commit": commit or None,
        "runner_git_dirty": None if status is None else bool(status),
    }


def runtime_version_provenance(*, include_ragas: bool) -> dict[str, Any]:
    """Capture only installed packages relevant to this evaluation mode."""

    distributions = ["note-rag"]
    if include_ragas:
        distributions.extend(("ragas", "google-genai"))
    package_versions: dict[str, str] = {}
    for distribution in distributions:
        try:
            package_versions[distribution] = importlib_metadata.version(
                distribution
            )
        except importlib_metadata.PackageNotFoundError:
            continue
    return {
        "python_version": platform.python_version(),
        "package_versions": package_versions,
    }


def _normalized_text_sha256(path: Path) -> str:
    """Hash decoded text after universal-newline normalization."""

    normalized = path.read_text(encoding="utf-8").encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def validate_gold_manifest(
    *,
    manifest_path: Path,
    dataset_path: Path,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the manifest against the exact selected dataset file."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"gold manifest is missing: {manifest_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"gold manifest is invalid JSON: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("gold manifest must be a JSON object")

    normalized_dataset_sha256 = _normalized_text_sha256(dataset_path)
    if manifest.get("dataset_sha256") != normalized_dataset_sha256:
        raise ValueError("gold manifest dataset hash does not match the dataset")
    question_count = manifest.get("question_count")
    if (
        not isinstance(question_count, int)
        or isinstance(question_count, bool)
        or question_count != len(entries)
    ):
        raise ValueError("gold manifest question count does not match the dataset")

    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise ValueError("gold manifest sources must be a list")
    manifest_source_ids = []
    for source in sources:
        source_id = source.get("source_id") if isinstance(source, dict) else None
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("gold manifest source IDs must be nonempty strings")
        manifest_source_ids.append(source_id)
    if len(set(manifest_source_ids)) != len(manifest_source_ids):
        raise ValueError("gold manifest source IDs must be unique")

    dataset_source_ids = []
    for entry in entries:
        passages = entry.get("relevant_passages")
        if not isinstance(passages, list):
            raise ValueError("dataset relevant_passages must be a list")
        for passage in passages:
            source_id = (
                passage.get("source_id") if isinstance(passage, dict) else None
            )
            if not isinstance(source_id, str) or not source_id:
                raise ValueError("dataset source IDs must be nonempty strings")
            dataset_source_ids.append(source_id)
    if set(manifest_source_ids) != set(dataset_source_ids):
        raise ValueError("gold manifest source IDs do not match the dataset")

    return {
        "gold_manifest": _safe_metadata_path(manifest_path),
        "gold_manifest_sha256": sha256_file(manifest_path),
        "dataset_sha256": normalized_dataset_sha256,
        "dataset_file_sha256": sha256_file(dataset_path),
        "gold_question_count": question_count,
        "gold_source_ids": sorted(set(manifest_source_ids)),
    }


def _manifest_source_hashes(manifest_path: Path) -> dict[str, str]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read gold manifest: {manifest_path}") from error
    sources = manifest.get("sources") if isinstance(manifest, dict) else None
    if not isinstance(sources, list):
        raise ValueError("gold manifest sources must be a list")
    expected: dict[str, str] = {}
    for source in sources:
        source_id = source.get("source_id") if isinstance(source, dict) else None
        source_hash = (
            source.get("source_sha256") if isinstance(source, dict) else None
        )
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("gold manifest source IDs must be nonempty strings")
        if not isinstance(source_hash, str) or not _SHA256_PATTERN.fullmatch(
            source_hash
        ):
            raise ValueError(f"gold manifest hash for {source_id!r} is invalid")
        if source_id in expected:
            raise ValueError("gold manifest source IDs must be unique")
        expected[source_id] = source_hash
    return expected


def sample_generation_provenance(
    samples: Sequence[Mapping[str, Any]],
    *,
    expected_answer_model: str | None = None,
) -> dict[str, Any]:
    """Summarize the answer models and exact prompt hashes in a sample set."""

    models = []
    prompt_hashes = []
    for sample in samples:
        model = sample.get("answer_model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"sample {sample.get('id')!r} answer model is invalid")
        prompt_hash = sample.get("generation_prompt_sha256")
        if not isinstance(prompt_hash, str) or not _SHA256_PATTERN.fullmatch(
            prompt_hash
        ):
            raise ValueError(
                f"sample {sample.get('id')!r} generation prompt hash is invalid"
            )
        models.append(model)
        prompt_hashes.append(prompt_hash)

    model_set = sorted(set(models))
    prompt_hash_set = sorted(set(prompt_hashes))
    if len(model_set) != 1:
        raise ValueError(
            "controlled generation requires exactly one answer model; "
            f"got {model_set!r}"
        )
    if expected_answer_model is not None and model_set != [expected_answer_model]:
        raise ValueError(
            "answer model set does not match --expect-answer-model: "
            f"expected {expected_answer_model!r}, got {model_set!r}"
        )
    return {
        "answer_models": model_set,
        "answer_model_count": len(model_set),
        "generation_prompt_sha256s": prompt_hash_set,
        "generation_prompt_sha256_count": len(prompt_hash_set),
    }


def capture_runtime_health(
    client: ApiClient,
    *,
    expected_chunking_strategy: str | None,
    expected_lexical_backend: str | None,
) -> dict[str, Any]:
    """Fetch one safe health snapshot and fail on controlled-run mismatches."""

    health = client.request("GET", "/health")
    if not isinstance(health, dict):
        raise RuntimeError("health endpoint returned an unexpected response")
    safe_health = {
        field: health.get(field)
        for field in (
            "status",
            "service",
            "version",
            "environment",
            "chunking_strategy",
            "lexical_backend",
            "embedding_model",
            "embedding_dimension",
            "reranker_model",
            "rerank_weight",
            "chat_model",
            "chat_temperature",
            "chat_max_output_tokens",
            "embedding_cache_enabled",
            "retrieval_cache_enabled",
            "corpus_version",
        )
        if isinstance(health.get(field), (str, bool, int, float))
    }
    expectations = {
        "chunking_strategy": expected_chunking_strategy,
        "lexical_backend": expected_lexical_backend,
    }
    mismatches = [
        f"{field}: expected {expected!r}, got {safe_health.get(field)!r}"
        for field, expected in expectations.items()
        if expected is not None and safe_health.get(field) != expected
    ]
    if mismatches:
        raise RuntimeError(
            "evaluation runtime configuration mismatch:\n"
            + "\n".join(f"- {mismatch}" for mismatch in mismatches)
        )
    corpus_version = safe_health.get("corpus_version")
    if (
        not isinstance(corpus_version, int)
        or isinstance(corpus_version, bool)
        or corpus_version < 0
    ):
        raise RuntimeError("health endpoint has no valid corpus version")
    return safe_health


def verify_corpus(
    client: ApiClient,
    *,
    manifest_path: Path,
) -> dict[str, Any]:
    """Verify the indexed corpus and return a compact, non-secret snapshot."""

    expected = _manifest_source_hashes(manifest_path)
    documents = client.request("GET", "/api/v1/documents")
    if not isinstance(documents, list) or not all(
        isinstance(document, dict) for document in documents
    ):
        raise RuntimeError("documents endpoint returned an unexpected response")
    by_filename: dict[str, list[dict[str, Any]]] = {}
    for document in documents:
        filename = document.get("filename")
        if isinstance(filename, str):
            by_filename.setdefault(filename, []).append(document)

    snapshots = []
    for source_id, source_hash in sorted(expected.items()):
        matches = by_filename.get(source_id, [])
        if len(matches) != 1:
            raise RuntimeError(
                f"benchmark corpus requires exactly one document named {source_id!r}"
            )
        document = matches[0]
        if document.get("content_hash") != source_hash:
            raise RuntimeError(f"benchmark document hash changed: {source_id}")
        if (
            document.get("status") != "ready"
            or document.get("indexing_status") != "indexed"
        ):
            raise RuntimeError(
                f"benchmark document is not ready and indexed: {source_id}"
            )
        snapshots.append(
            {
                field: document.get(field)
                for field in (
                    "id",
                    "filename",
                    "content_hash",
                    "status",
                    "indexing_status",
                    "embedding_model",
                    "chunk_count",
                    "token_count",
                )
                if isinstance(document.get(field), (str, bool, int, float))
            }
        )
    return {
        "corpus_document_count": len(snapshots),
        "corpus_documents": snapshots,
    }


def collect_samples(
    *,
    client: ApiClient,
    entries: Sequence[Mapping[str, Any]],
    mode: str,
    candidate_k: int,
    max_chunks: int,
    max_context_tokens: int,
    vector_weight: float,
    rerank: bool,
    rerank_weight: float | None,
    delay_ms: float = 0,
    expected_chunking_strategy: str | None = None,
    expected_lexical_backend: str | None = None,
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate answers and capture contexts under one fixed configuration."""

    health_before = capture_runtime_health(
        client,
        expected_chunking_strategy=expected_chunking_strategy,
        expected_lexical_backend=expected_lexical_backend,
    )
    corpus_provenance = verify_corpus(client, manifest_path=manifest_path)
    corpus_version = health_before["corpus_version"]
    filenames = sorted(
        {
            passage["source_id"]
            for entry in entries
            for passage in entry["relevant_passages"]
        }
    )
    options: dict[str, Any] = {
        "mode": mode,
        "candidate_k": candidate_k,
        "max_chunks": max_chunks,
        "max_context_tokens": max_context_tokens,
        "vector_weight": vector_weight,
        "rerank": rerank,
        "filters": {"filenames": filenames},
    }
    if rerank_weight is not None:
        options["rerank_weight"] = rerank_weight

    samples = []
    for number, entry in enumerate(entries, start=1):
        payload = {
            "query": entry["query"],
            "include_generation_context": True,
            **options,
        }
        chat_started = time.perf_counter()
        chat = client.request("POST", "/api/v1/chat", payload)
        chat_latency_ms = (time.perf_counter() - chat_started) * 1000

        if not isinstance(chat, dict):
            raise RuntimeError("chat endpoint returned an unexpected response")
        context = chat.get("generation_context")
        if not isinstance(context, dict):
            raise RuntimeError("chat response has no generation context")
        chunks = context.get("chunks")
        if not isinstance(chunks, list):
            raise RuntimeError("generation context has no chunk list")
        if not all(isinstance(chunk, dict) for chunk in chunks):
            raise RuntimeError("generation context chunks must be objects")
        samples.append(
            {
                "id": entry["id"],
                "user_input": entry["query"],
                "response": chat.get("answer"),
                "reference": entry["expected_answer"],
                "retrieved_contexts": [chunk.get("text") for chunk in chunks],
                "tags": entry.get("tags", []),
                "answer_model": chat.get("model_name"),
                "conversation_id": chat.get("conversation_id"),
                "citations": chat.get("citations", []),
                "context_chunk_ids": [chunk.get("chunk_id") for chunk in chunks],
                "context_citation_ids": [
                    chunk.get("citation_id") for chunk in chunks
                ],
                "rendered_context": context.get("context"),
                "generation_prompt_sha256": chat.get(
                    "generation_prompt_sha256"
                ),
                "context_token_count": context.get("token_count"),
                "context_truncated": context.get("truncated"),
                "context_corpus_version": context.get("corpus_version"),
                "chat_latency_ms": chat_latency_ms,
                "total_latency_ms": chat_latency_ms,
            }
        )
        print(
            f"[{number:02}/{len(entries)}] {entry['id']}: "
            f"answer + {len(chunks)} contexts ({chat_latency_ms:.1f} ms)"
        )
        if delay_ms and number < len(entries):
            time.sleep(delay_ms / 1000)
    validate_samples(
        samples,
        entries,
        expected_corpus_version=corpus_version,
    )
    health_after = capture_runtime_health(
        client,
        expected_chunking_strategy=expected_chunking_strategy,
        expected_lexical_backend=expected_lexical_backend,
    )
    if health_after["corpus_version"] != corpus_version:
        raise RuntimeError("corpus version changed during answer collection")
    return samples, {
        "runtime_health_before": health_before,
        "runtime_health_after": health_after,
        "collection_corpus_version": corpus_version,
        **corpus_provenance,
    }


def validate_samples(
    samples: Sequence[Mapping[str, Any]],
    entries: Sequence[Mapping[str, Any]],
    *,
    expected_corpus_version: int | None = None,
) -> None:
    """Ensure saved samples correspond exactly to the selected gold entries."""

    expected_ids = []
    for entry in entries:
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id:
            raise ValueError("dataset IDs must be nonempty strings")
        expected_ids.append(entry_id)
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("dataset IDs must be unique")
    actual_ids = []
    for sample in samples:
        sample_id = sample.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("sample IDs must be nonempty strings")
        actual_ids.append(sample_id)
    if len(set(actual_ids)) != len(actual_ids):
        raise ValueError("sample IDs must be unique")
    if actual_ids != expected_ids:
        raise ValueError("sample IDs and order do not match the selected dataset")
    required = {
        "user_input",
        "response",
        "reference",
        "retrieved_contexts",
        "answer_model",
        "citations",
        "context_chunk_ids",
        "context_citation_ids",
        "rendered_context",
        "generation_prompt_sha256",
        "context_token_count",
        "context_truncated",
        "context_corpus_version",
    }
    for sample, entry in zip(samples, entries, strict=True):
        missing = sorted(required - sample.keys())
        if missing:
            rendered = ", ".join(missing)
            raise ValueError(f"sample {sample.get('id')!r} is missing: {rendered}")
        if not isinstance(sample["user_input"], str):
            raise ValueError(f"sample {sample['id']!r} query must be a string")
        if sample["user_input"] != entry.get("query"):
            raise ValueError(f"sample {sample['id']!r} query does not match dataset")
        if not isinstance(sample["reference"], str):
            raise ValueError(f"sample {sample['id']!r} reference must be a string")
        if sample["reference"] != entry.get("expected_answer"):
            raise ValueError(
                f"sample {sample['id']!r} reference does not match dataset"
            )
        response = sample["response"]
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"sample {sample['id']!r} response must be nonempty")
        answer_model = sample["answer_model"]
        if not isinstance(answer_model, str) or not answer_model.strip():
            raise ValueError(
                f"sample {sample['id']!r} answer model must be nonempty"
            )
        contexts = sample["retrieved_contexts"]
        chunk_ids = sample["context_chunk_ids"]
        citation_ids = sample["context_citation_ids"]
        if not isinstance(contexts, list) or not all(
            isinstance(context, str) for context in contexts
        ):
            raise ValueError(f"sample {sample['id']!r} contexts must be strings")
        if not isinstance(chunk_ids, list) or not all(
            isinstance(chunk_id, str) and bool(chunk_id) for chunk_id in chunk_ids
        ):
            raise ValueError(
                f"sample {sample['id']!r} context IDs must be nonempty strings"
            )
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError(f"sample {sample['id']!r} context IDs must be unique")
        if not isinstance(citation_ids, list) or not all(
            isinstance(citation_id, int)
            and not isinstance(citation_id, bool)
            and citation_id > 0
            for citation_id in citation_ids
        ):
            raise ValueError(
                f"sample {sample['id']!r} citation IDs must be positive integers"
            )
        if len(set(citation_ids)) != len(citation_ids):
            raise ValueError(f"sample {sample['id']!r} citation IDs must be unique")
        if not (
            len(chunk_ids) == len(contexts) == len(citation_ids)
        ):
            raise ValueError(
                f"sample {sample['id']!r} context fields are not aligned"
            )
        citations = sample["citations"]
        if not isinstance(citations, list):
            raise ValueError(f"sample {sample['id']!r} citations must be a list")
        cited_ids = []
        for citation in citations:
            citation_id = (
                citation.get("citation_id")
                if isinstance(citation, dict)
                else None
            )
            if (
                not isinstance(citation_id, int)
                or isinstance(citation_id, bool)
                or citation_id <= 0
            ):
                raise ValueError(
                    f"sample {sample['id']!r} cited IDs must be positive integers"
                )
            cited_ids.append(citation_id)
        if len(set(cited_ids)) != len(cited_ids):
            raise ValueError(f"sample {sample['id']!r} cited IDs must be unique")
        if not set(cited_ids).issubset(citation_ids):
            raise ValueError(
                f"sample {sample['id']!r} cited IDs are absent from context"
            )
        if not isinstance(sample["rendered_context"], str):
            raise ValueError(
                f"sample {sample['id']!r} rendered context must be a string"
            )
        token_count = sample["context_token_count"]
        if (
            not isinstance(token_count, int)
            or isinstance(token_count, bool)
            or token_count < 0
        ):
            raise ValueError(
                f"sample {sample['id']!r} context token count must be a "
                "nonnegative integer"
            )
        if not isinstance(sample["context_truncated"], bool):
            raise ValueError(
                f"sample {sample['id']!r} context truncated must be boolean"
            )
        corpus_version = sample["context_corpus_version"]
        if (
            not isinstance(corpus_version, int)
            or isinstance(corpus_version, bool)
            or corpus_version < 0
        ):
            raise ValueError(
                f"sample {sample['id']!r} context corpus version must be a "
                "nonnegative integer"
            )
        if (
            expected_corpus_version is not None
            and corpus_version != expected_corpus_version
        ):
            raise ValueError(
                f"sample {sample['id']!r} context corpus version does not "
                "match the run"
            )
        prompt_hash = sample["generation_prompt_sha256"]
        if not isinstance(prompt_hash, str) or not _SHA256_PATTERN.fullmatch(
            prompt_hash
        ):
            raise ValueError(
                f"sample {sample['id']!r} generation prompt hash is invalid"
            )
    corpus_versions = {sample["context_corpus_version"] for sample in samples}
    if len(corpus_versions) != 1:
        raise ValueError("samples contain multiple context corpus versions")


def companion_summary_path(samples_path: Path) -> Path:
    suffix = ".results.jsonl"
    if not samples_path.name.endswith(suffix):
        raise ValueError("sample path must end with '.results.jsonl'")
    return samples_path.with_name(
        samples_path.name.removesuffix(suffix) + ".summary.json"
    )


def load_and_validate_sample_provenance(
    *,
    samples_path: Path,
    dataset_sha256: str,
    configuration: Mapping[str, Any],
    sample_count: int,
) -> dict[str, Any]:
    """Verify a replay bundle before any evaluator calls are made."""

    summary_path = companion_summary_path(samples_path)
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"sample summary is missing: {summary_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"sample summary is invalid JSON: {summary_path}") from error
    metadata = summary.get("metadata") if isinstance(summary, dict) else None
    if not isinstance(metadata, dict):
        raise ValueError("sample summary has no metadata object")
    actual_hash = sha256_file(samples_path)
    if metadata.get("results_sha256") != actual_hash:
        raise ValueError("sample JSONL hash does not match its summary")
    if metadata.get("dataset_sha256") != dataset_sha256:
        raise ValueError("sample dataset hash does not match the selected dataset")
    if metadata.get("sample_count") != sample_count:
        raise ValueError("sample count does not match its summary")
    mismatches = [
        field
        for field in _PRODUCER_CONTROL_FIELDS
        if metadata.get(field) != configuration.get(field)
    ]
    if mismatches:
        raise ValueError(
            "sample producer configuration differs for: "
            + ", ".join(mismatches)
        )
    return {
        **metadata,
        "source_samples_sha256": actual_hash,
        "source_samples_summary_sha256": sha256_file(summary_path),
    }


def _validate_generation_provenance(
    producer_metadata: Mapping[str, Any],
    generation_provenance: Mapping[str, Any],
) -> None:
    mismatches = [
        field
        for field, value in generation_provenance.items()
        if producer_metadata.get(field) != value
    ]
    if mismatches:
        raise ValueError(
            "sample generation provenance differs for: "
            + ", ".join(mismatches)
        )


def load_replay_bundle(
    *,
    samples_path: Path,
    dataset_sha256: str,
    configuration: Mapping[str, Any],
    limit: int | None,
    expected_answer_model: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    """Validate a complete replay bundle before selecting an evaluation prefix."""

    source_samples = load_jsonl(samples_path)
    source_sample_count = len(source_samples)
    producer_metadata = load_and_validate_sample_provenance(
        samples_path=samples_path,
        dataset_sha256=dataset_sha256,
        configuration=configuration,
        sample_count=source_sample_count,
    )
    generation_provenance = sample_generation_provenance(
        source_samples,
        expected_answer_model=expected_answer_model,
    )
    _validate_generation_provenance(producer_metadata, generation_provenance)
    evaluated_samples = source_samples[:limit] if limit is not None else source_samples
    return evaluated_samples, producer_metadata, source_sample_count


def evaluation_scope_provenance(
    samples: Sequence[Mapping[str, Any]],
    *,
    source_sample_count: int,
) -> dict[str, Any]:
    """Describe exactly which prefix of a hash-bound source bundle was scored."""

    return {
        "source_sample_count": source_sample_count,
        "evaluated_sample_count": len(samples),
        "evaluated_sample_ids": [sample["id"] for sample in samples],
    }


def validate_score(name: str, value: Any) -> float:
    """Fail closed when an evaluator returns an unusable probability score."""

    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} score is not numeric") from error
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"{name} score must be finite and within [0, 1]")
    return score


def evaluate_deterministic_samples(
    samples: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    results = []
    for sample in samples:
        if "context_citation_ids" not in sample:
            raise ValueError(
                f"sample {sample.get('id')!r} is missing: context_citation_ids"
            )
        citation_ids = sample["context_citation_ids"]
        if not isinstance(citation_ids, list):
            raise ValueError(
                f"sample {sample.get('id')!r} citation IDs must be a list"
            )
        scores = evaluate_deterministic_answer_quality(
            response=sample["response"],
            reference=sample["reference"],
            retrieved_contexts=sample["retrieved_contexts"],
            available_citation_ids=set(citation_ids),
        )
        results.append({**sample, "deterministic": scores})
    return results


def summarize_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {"sample_count": len(samples)}
    for field, prefix in (
        ("chat_latency_ms", "chat_latency"),
        ("total_latency_ms", "total_latency"),
    ):
        values = [
            float(sample[field])
            for sample in samples
            if sample.get(field) is not None
        ]
        metrics.update(summarize_latencies(values, prefix=prefix))
    token_counts = [
        float(sample["context_token_count"])
        for sample in samples
        if sample.get("context_token_count") is not None
    ]
    metrics["context_tokens_mean"] = (
        statistics.fmean(token_counts) if token_counts else 0.0
    )
    metrics["context_truncated_count"] = sum(
        sample.get("context_truncated") is True for sample in samples
    )
    return metrics


def create_scorers(
    *,
    metric_names: Sequence[str],
    api_key: str,
    evaluator_model: str,
    embedding_model: str,
) -> dict[str, RagasScorer]:
    """Create modern RAGAS collection metrics backed by Google Gemini."""

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required")
    try:
        from google import genai
        from ragas.embeddings import GoogleEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerCorrectness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as error:
        raise RuntimeError(
            "RAGAS dependencies are missing; install with "
            'pip install -e ".[evaluation]"'
        ) from error

    client = genai.Client(api_key=api_key)
    llm = llm_factory(evaluator_model, provider="google", client=client)
    embeddings = GoogleEmbeddings(client=client, model=embedding_model)
    factories = {
        "faithfulness": lambda: Faithfulness(llm=llm),
        "answer_relevancy": lambda: AnswerRelevancy(llm=llm, embeddings=embeddings),
        "answer_correctness": lambda: AnswerCorrectness(llm=llm, embeddings=embeddings),
        "context_precision": lambda: ContextPrecision(llm=llm),
        "context_recall": lambda: ContextRecall(llm=llm),
    }
    return {name: factories[name]() for name in metric_names}


def evaluate_samples(
    samples: Sequence[Mapping[str, Any]],
    scorers: Mapping[str, RagasScorer],
) -> list[dict[str, Any]]:
    """Score samples while retaining per-metric judge reasoning."""

    metric_inputs = {
        "faithfulness": ("user_input", "response", "retrieved_contexts"),
        "answer_relevancy": ("user_input", "response"),
        "answer_correctness": ("response", "reference"),
        "context_precision": ("user_input", "reference", "retrieved_contexts"),
        "context_recall": ("user_input", "reference", "retrieved_contexts"),
    }
    results = []
    for number, sample in enumerate(samples, start=1):
        scores: dict[str, float] = {}
        reasons: dict[str, Any] = {}
        for name, scorer in scorers.items():
            kwargs = {key: sample[key] for key in metric_inputs[name]}
            result = scorer.score(**kwargs)
            scores[name] = validate_score(name, result.value)
            reason = getattr(result, "reason", None)
            if reason is not None:
                reasons[name] = reason
        results.append({**sample, "ragas": scores, "ragas_reasons": reasons})
        rendered = ", ".join(f"{name}={value:.3f}" for name, value in scores.items())
        print(f"[{number:02}/{len(samples)}] {sample['id']}: {rendered}")
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect Note RAG answers and optionally judge them with RAGAS."
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPOSITORY_ROOT / "benchmarks/gold/dataset.jsonl",
    )
    parser.add_argument(
        "--gold-manifest",
        type=Path,
        default=REPOSITORY_ROOT / "benchmarks/gold/manifest.json",
    )
    parser.add_argument("--samples", type=Path, help="judge an existing sample JSONL")
    parser.add_argument(
        "--output-dir", type=Path, default=REPOSITORY_ROOT / "tmp/benchmarks"
    )
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument(
        "--evaluator",
        choices=("deterministic", "ragas-google"),
        default="deterministic",
    )
    parser.add_argument("--require-full-dataset", action="store_true")
    parser.add_argument("--expected-count", type=int, default=48)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--mode",
        choices=("vector", "keyword", "hybrid"),
        default="hybrid",
    )
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--max-chunks", type=int, default=8)
    parser.add_argument("--max-context-tokens", type=int, default=1800)
    parser.add_argument("--vector-weight", type=float, default=0.7)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--rerank-weight", type=float)
    parser.add_argument("--delay-ms", type=float, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--api-token-env", default="API_AUTH_TOKEN")
    parser.add_argument(
        "--expect-chunking-strategy",
        choices=("fixed", "recursive"),
    )
    parser.add_argument(
        "--expect-lexical-backend",
        choices=("bm25", "postgres_fts"),
    )
    parser.add_argument("--expect-answer-model")
    parser.add_argument(
        "--evaluator-model",
        default=os.getenv("RAGAS_EVALUATOR_MODEL", "gemini-2.5-flash"),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("RAGAS_EMBEDDING_MODEL", "gemini-embedding-001"),
    )
    parser.add_argument(
        "--metrics", nargs="+", choices=_METRIC_NAMES, default=list(_METRIC_NAMES)
    )
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        validate_run_label(args.label)
    except ValueError as error:
        parser.error(str(error))
    if args.samples is not None and args.collect_only:
        parser.error("--samples and --collect-only cannot be used together")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")
    if args.expected_count <= 0:
        parser.error("--expected-count must be greater than zero")
    if args.require_full_dataset and args.limit is not None:
        parser.error("--require-full-dataset cannot be combined with --limit")
    if not 1 <= args.candidate_k <= 100:
        parser.error("--candidate-k must be between 1 and 100")
    if not 1 <= args.max_chunks <= 100:
        parser.error("--max-chunks must be between 1 and 100")
    if args.max_context_tokens <= 0:
        parser.error("--max-context-tokens must be greater than zero")
    if not 0 <= args.vector_weight <= 1:
        parser.error("--vector-weight must be between 0 and 1")
    if args.rerank_weight is not None and not 0 <= args.rerank_weight <= 1:
        parser.error("--rerank-weight must be between 0 and 1")
    if args.delay_ms < 0:
        parser.error("--delay-ms cannot be negative")

    dataset_path = args.dataset.resolve()
    full_entries = load_jsonl(dataset_path)
    if args.require_full_dataset and len(full_entries) != args.expected_count:
        parser.error(
            f"full dataset must contain exactly {args.expected_count} entries"
        )
    entries = full_entries[: args.limit] if args.limit is not None else full_entries
    configuration = {
        "base_url": args.base_url.rstrip("/"),
        "mode": args.mode,
        "candidate_k": args.candidate_k,
        "max_chunks": args.max_chunks,
        "max_context_tokens": args.max_context_tokens,
        "vector_weight": args.vector_weight,
        "rerank": not args.no_rerank,
        "rerank_weight": args.rerank_weight,
        "expected_chunking_strategy": args.expect_chunking_strategy,
        "expected_lexical_backend": args.expect_lexical_backend,
        "expected_answer_model": args.expect_answer_model,
    }
    metadata = build_run_metadata(
        label=args.label, dataset_path=dataset_path, configuration=configuration
    )
    metadata["dataset"] = _safe_metadata_path(dataset_path)
    metadata.update(
        validate_gold_manifest(
            manifest_path=args.gold_manifest.resolve(),
            dataset_path=dataset_path,
            entries=full_entries,
        )
    )
    metadata.update(runner_git_provenance())
    metadata.update(runtime_version_provenance(include_ragas=False))
    output_dir = args.output_dir.resolve()

    if args.samples is not None:
        samples_path = args.samples.resolve()
        samples, producer_metadata, source_sample_count = load_replay_bundle(
            samples_path=samples_path,
            dataset_sha256=str(metadata["dataset_sha256"]),
            configuration=configuration,
            limit=args.limit,
            expected_answer_model=args.expect_answer_model,
        )
    else:
        token = os.getenv(args.api_token_env, "") if args.api_token_env else ""
        client = ApiClient(args.base_url, token=token, timeout=args.timeout_seconds)
        samples, runtime_provenance = collect_samples(
            client=client,
            entries=entries,
            mode=args.mode,
            candidate_k=args.candidate_k,
            max_chunks=args.max_chunks,
            max_context_tokens=args.max_context_tokens,
            vector_weight=args.vector_weight,
            rerank=not args.no_rerank,
            rerank_weight=args.rerank_weight,
            delay_ms=args.delay_ms,
            expected_chunking_strategy=args.expect_chunking_strategy,
            expected_lexical_backend=args.expect_lexical_backend,
            manifest_path=args.gold_manifest.resolve(),
        )
        runtime_answer_model = runtime_provenance["runtime_health_before"].get(
            "chat_model"
        )
        controlled_answer_model = (
            args.expect_answer_model
            if args.expect_answer_model is not None
            else runtime_answer_model
        )
        generation_provenance = sample_generation_provenance(
            samples,
            expected_answer_model=controlled_answer_model,
        )
        sample_paths = write_evaluation_artifacts(
            output_dir=output_dir,
            label=args.label,
            kind="rag-samples",
            results=samples,
            metadata={
                **metadata,
                **runtime_provenance,
                **generation_provenance,
                "sample_count": len(samples),
            },
            metrics=summarize_samples(samples),
        )
        print("Wrote samples:")
        for path in sample_paths:
            print(f"- {path}")
        producer_metadata = load_and_validate_sample_provenance(
            samples_path=sample_paths[0],
            dataset_sha256=str(metadata["dataset_sha256"]),
            configuration=configuration,
            sample_count=len(samples),
        )
        source_sample_count = len(samples)

    producer_corpus_version = producer_metadata.get("collection_corpus_version")
    if (
        not isinstance(producer_corpus_version, int)
        or isinstance(producer_corpus_version, bool)
        or producer_corpus_version < 0
    ):
        raise ValueError("sample producer has no valid collection corpus version")
    validate_samples(
        samples,
        entries,
        expected_corpus_version=producer_corpus_version,
    )
    generation_provenance = sample_generation_provenance(
        samples,
        expected_answer_model=args.expect_answer_model,
    )
    if args.samples is None:
        _validate_generation_provenance(
            producer_metadata,
            generation_provenance,
        )
    if args.require_full_dataset and len(samples) != args.expected_count:
        parser.error(
            f"full evaluation requires exactly {args.expected_count} samples"
        )
    if args.collect_only:
        return

    evaluation_scope = evaluation_scope_provenance(
        samples,
        source_sample_count=source_sample_count,
    )
    evaluator_environment = {
        **runner_git_provenance(),
        **runtime_version_provenance(
            include_ragas=args.evaluator == "ragas-google"
        ),
    }

    if args.evaluator == "deterministic":
        results = evaluate_deterministic_samples(samples)
        metrics = aggregate_numeric_scores(
            [result["deterministic"] for result in results]
        )
        paths = write_evaluation_artifacts(
            output_dir=output_dir,
            label=args.label,
            kind="deterministic-quality",
            results=results,
            metadata={
                **producer_metadata,
                "sample_count": len(results),
                "evaluator": "deterministic-lexical-proxies-v1",
                "evaluator_environment": evaluator_environment,
                **evaluation_scope,
            },
            metrics=metrics,
        )
        print(json.dumps(metrics, indent=2))
        print("Wrote deterministic proxy results:")
        for path in paths:
            print(f"- {path}")
        return

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    scorers = create_scorers(
        metric_names=args.metrics,
        api_key=api_key,
        evaluator_model=args.evaluator_model,
        embedding_model=args.embedding_model,
    )
    results = evaluate_samples(samples, scorers)
    metrics = aggregate_ragas_scores([result["ragas"] for result in results])
    paths = write_evaluation_artifacts(
        output_dir=output_dir,
        label=args.label,
        kind="ragas",
        results=results,
        metadata={
            **producer_metadata,
            "sample_count": len(results),
            "metrics": list(args.metrics),
            "evaluator_provider": "google",
            "evaluator_model": args.evaluator_model,
            "embedding_model": args.embedding_model,
            "context_input": "chunk bodies only",
            "evaluator_environment": evaluator_environment,
            **evaluation_scope,
        },
        metrics=metrics,
    )
    print(json.dumps(metrics, indent=2))
    print("Wrote RAGAS results:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()

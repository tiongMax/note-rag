from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from note_rag.evaluation import aggregate_ragas_scores


def _load_runner() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "run_ragas_evaluation.py"
    spec = importlib.util.spec_from_file_location("ragas_runner", path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def _valid_sample(
    sample_id: str = "q1",
    *,
    query: str = "question",
    reference: str = "reference",
) -> dict[str, Any]:
    return {
        "id": sample_id,
        "user_input": query,
        "response": "supported answer [1]",
        "reference": reference,
        "retrieved_contexts": ["supported answer"],
        "answer_model": "fake-chat",
        "citations": [{"citation_id": 1}],
        "context_chunk_ids": ["chunk-1"],
        "context_citation_ids": [1],
        "rendered_context": "[1] Source: source.pdf\nsupported answer",
        "generation_prompt_sha256": "a" * 64,
        "context_token_count": 2,
        "context_truncated": False,
        "context_corpus_version": 7,
    }


def test_aggregates_ragas_scores() -> None:
    summary = aggregate_ragas_scores(
        [
            {"faithfulness": 1.0, "context_recall": 0.5},
            {"faithfulness": 0.5, "context_recall": 1.0},
        ]
    )

    assert summary == {
        "sample_count": 2,
        "faithfulness": 0.75,
        "context_recall": 0.75,
    }


def test_rejects_inconsistent_ragas_metrics() -> None:
    with pytest.raises(ValueError, match="same metrics"):
        aggregate_ragas_scores([{"faithfulness": 1.0}, {"context_recall": 1.0}])


class FakeScorer:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[dict[str, object]] = []

    def score(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(value=self.value, reason="test reason")


def test_runner_maps_samples_to_modern_ragas_metric_inputs() -> None:
    runner = _load_runner()
    faithfulness = FakeScorer(0.8)
    precision = FakeScorer(0.6)

    results = runner.evaluate_samples(
        [
            {
                "id": "q1",
                "user_input": "question",
                "response": "answer",
                "reference": "reference",
                "retrieved_contexts": ["context"],
            }
        ],
        {"faithfulness": faithfulness, "context_precision": precision},
    )

    assert faithfulness.calls == [
        {
            "user_input": "question",
            "response": "answer",
            "retrieved_contexts": ["context"],
        }
    ]
    assert precision.calls == [
        {
            "user_input": "question",
            "reference": "reference",
            "retrieved_contexts": ["context"],
        }
    ]
    assert results[0]["ragas"] == {
        "faithfulness": 0.8,
        "context_precision": 0.6,
    }
    assert results[0]["ragas_reasons"] == {
        "faithfulness": "test reason",
        "context_precision": "test reason",
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_runner_rejects_invalid_judge_scores(value: float) -> None:
    runner = _load_runner()

    with pytest.raises(ValueError, match=r"finite and within \[0, 1\]"):
        runner.validate_score("faithfulness", value)


class FakeClient:
    def __init__(
        self,
        *,
        corpus_versions: tuple[int, int] = (7, 7),
        document_hash: str = "f" * 64,
    ) -> None:
        self.health_requests = 0
        self.document_requests = 0
        self.corpus_versions = corpus_versions
        self.document_hash = document_hash

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        if method == "GET" and path == "/health":
            corpus_version = self.corpus_versions[
                min(self.health_requests, len(self.corpus_versions) - 1)
            ]
            self.health_requests += 1
            return {
                "status": "ok",
                "service": "note-rag",
                "version": "0.1.0",
                "environment": "test",
                "chunking_strategy": "recursive",
                "lexical_backend": "bm25",
                "embedding_model": "fake-embedding",
                "embedding_dimension": 8,
                "reranker_model": "fake-reranker",
                "rerank_weight": 0.7,
                "chat_model": "fake-chat",
                "chat_temperature": 0.1,
                "chat_max_output_tokens": 128,
                "embedding_cache_enabled": True,
                "retrieval_cache_enabled": True,
                "corpus_version": corpus_version,
                "ignored_secret": "must-not-be-captured",
            }
        if method == "GET" and path == "/api/v1/documents":
            self.document_requests += 1
            return [
                {
                    "id": "document-1",
                    "filename": "source.pdf",
                    "content_hash": self.document_hash,
                    "status": "ready",
                    "indexing_status": "indexed",
                    "embedding_model": "fake-embedding",
                    "chunk_count": 2,
                    "token_count": 20,
                    "storage_uri": "secret-path-must-not-be-captured",
                }
            ]
        assert method == "POST"
        assert path == "/api/v1/chat"
        assert payload is not None
        assert payload["filters"] == {"filenames": ["source.pdf"]}
        assert payload["include_generation_context"] is True
        return {
            "answer": "supported answer [1]",
            "model_name": "fake-chat",
            "conversation_id": "conversation-1",
            "citations": [{"citation_id": 1, "chunk_id": "chunk-1"}],
            "generation_prompt_sha256": "a" * 64,
            "generation_context": {
                "chunks": [
                    {
                        "citation_id": 1,
                        "chunk_id": "chunk-1",
                        "text": "support",
                    }
                ],
                "context": "[1] Source: source.pdf\nsupport",
                "token_count": 1,
                "truncated": False,
                "corpus_version": self.corpus_versions[0],
            },
        }


def test_collects_replayable_samples_and_latency_metrics(tmp_path: Path) -> None:
    runner = _load_runner()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sources": [
                    {"source_id": "source.pdf", "source_sha256": "f" * 64}
                ]
            }
        ),
        encoding="utf-8",
    )
    entries = [
        {
            "id": "q1",
            "query": "question",
            "expected_answer": "answer",
            "relevant_passages": [{"source_id": "source.pdf"}],
        }
    ]

    client = FakeClient()
    samples, runtime_provenance = runner.collect_samples(
        client=client,
        entries=entries,
        mode="hybrid",
        candidate_k=20,
        max_chunks=8,
        max_context_tokens=1200,
        vector_weight=0.7,
        rerank=True,
        rerank_weight=None,
        expected_chunking_strategy="recursive",
        expected_lexical_backend="bm25",
        manifest_path=manifest,
    )
    runner.validate_samples(samples, entries)
    summary = runner.summarize_samples(samples)

    assert samples[0]["response"] == "supported answer [1]"
    assert samples[0]["retrieved_contexts"] == ["support"]
    assert samples[0]["rendered_context"].endswith("support")
    assert summary["sample_count"] == 1
    assert summary["chat_latency_count"] == 1
    assert summary["context_tokens_mean"] == 1
    assert samples[0]["context_corpus_version"] == 7
    assert client.health_requests == 2
    assert client.document_requests == 1
    assert runtime_provenance["runtime_health_before"] == {
        "status": "ok",
        "service": "note-rag",
        "version": "0.1.0",
        "environment": "test",
        "chunking_strategy": "recursive",
        "lexical_backend": "bm25",
        "embedding_model": "fake-embedding",
        "embedding_dimension": 8,
        "reranker_model": "fake-reranker",
        "rerank_weight": 0.7,
        "chat_model": "fake-chat",
        "chat_temperature": 0.1,
        "chat_max_output_tokens": 128,
        "embedding_cache_enabled": True,
        "retrieval_cache_enabled": True,
        "corpus_version": 7,
    }
    assert runtime_provenance["runtime_health_after"]["corpus_version"] == 7
    assert runtime_provenance["collection_corpus_version"] == 7
    assert runtime_provenance["corpus_document_count"] == 1
    snapshot = runtime_provenance["corpus_documents"][0]
    assert snapshot["content_hash"] == "f" * 64
    assert "storage_uri" not in snapshot


def test_runtime_health_rejects_control_mismatch() -> None:
    runner = _load_runner()

    with pytest.raises(RuntimeError, match="runtime configuration mismatch"):
        runner.capture_runtime_health(
            FakeClient(),
            expected_chunking_strategy="fixed",
            expected_lexical_backend="bm25",
        )


def test_summarizes_answer_models_and_prompt_hashes() -> None:
    runner = _load_runner()
    samples = [
        {
            "id": "q1",
            "answer_model": "answer-v1",
            "generation_prompt_sha256": "b" * 64,
        },
        {
            "id": "q2",
            "answer_model": "answer-v1",
            "generation_prompt_sha256": "a" * 64,
        },
    ]

    provenance = runner.sample_generation_provenance(
        samples,
        expected_answer_model="answer-v1",
    )

    assert provenance == {
        "answer_models": ["answer-v1"],
        "answer_model_count": 1,
        "generation_prompt_sha256s": ["a" * 64, "b" * 64],
        "generation_prompt_sha256_count": 2,
    }
    with pytest.raises(ValueError, match="expect-answer-model"):
        runner.sample_generation_provenance(
            samples,
            expected_answer_model="answer-v2",
        )

    mixed = [
        samples[0],
        {**samples[1], "answer_model": "answer-v2"},
    ]
    with pytest.raises(ValueError, match="exactly one answer model"):
        runner.sample_generation_provenance(mixed)


def test_git_provenance_is_safe_and_degrades_outside_git(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner()

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        del kwargs
        if command[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="abc123\n")
        return SimpleNamespace(stdout=" M tracked.py\n")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner.runner_git_provenance(tmp_path) == {
        "runner_git_commit": "abc123",
        "runner_git_dirty": True,
    }

    def missing_git(command: list[str], **kwargs: object) -> SimpleNamespace:
        del command, kwargs
        raise FileNotFoundError

    monkeypatch.setattr(runner.subprocess, "run", missing_git)
    assert runner.runner_git_provenance(tmp_path) == {
        "runner_git_commit": None,
        "runner_git_dirty": None,
    }


def test_runtime_versions_include_optional_judge_packages_only_when_relevant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    versions = {
        "note-rag": "0.1.0",
        "ragas": "0.4.2",
        "google-genai": "2.12.0",
    }
    monkeypatch.setattr(
        runner.importlib_metadata,
        "version",
        lambda distribution: versions[distribution],
    )
    monkeypatch.setattr(runner.platform, "python_version", lambda: "3.12.9")

    deterministic = runner.runtime_version_provenance(include_ragas=False)
    judged = runner.runtime_version_provenance(include_ragas=True)

    assert deterministic == {
        "python_version": "3.12.9",
        "package_versions": {"note-rag": "0.1.0"},
    }
    assert judged["package_versions"] == versions


def test_gold_manifest_validates_normalized_hash_count_and_sources(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_bytes(b'{"id":"q1"}\r\n')
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_sha256": runner._normalized_text_sha256(dataset),
                "question_count": 1,
                "sources": [
                    {"source_id": "source.pdf", "source_sha256": "f" * 64}
                ],
            }
        ),
        encoding="utf-8",
    )
    entries = [
        {
            "id": "q1",
            "relevant_passages": [{"source_id": "source.pdf"}],
        }
    ]

    provenance = runner.validate_gold_manifest(
        manifest_path=manifest,
        dataset_path=dataset,
        entries=entries,
    )

    assert provenance["gold_manifest"] == "manifest.json"
    assert provenance["gold_manifest_sha256"] == runner.sha256_file(manifest)
    assert provenance["dataset_sha256"] == runner._normalized_text_sha256(dataset)
    assert provenance["dataset_file_sha256"] == runner.sha256_file(dataset)
    assert provenance["dataset_sha256"] != provenance["dataset_file_sha256"]
    assert provenance["gold_question_count"] == 1
    assert provenance["gold_source_ids"] == ["source.pdf"]

    changed_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    changed_manifest["question_count"] = 2
    manifest.write_text(json.dumps(changed_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="question count"):
        runner.validate_gold_manifest(
            manifest_path=manifest,
            dataset_path=dataset,
            entries=entries,
        )


def test_rejects_samples_from_a_different_dataset() -> None:
    runner = _load_runner()
    with pytest.raises(ValueError, match="IDs and order"):
        runner.validate_samples([{"id": "wrong"}], [{"id": "expected"}])


def test_rejects_replayed_sample_with_changed_query_or_reference() -> None:
    runner = _load_runner()
    sample = _valid_sample(query="changed")
    entry = {"id": "q1", "query": "question", "expected_answer": "reference"}

    with pytest.raises(ValueError, match="query does not match"):
        runner.validate_samples([sample], [entry])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("response", "", "response must be nonempty"),
        ("answer_model", "", "answer model must be nonempty"),
        ("retrieved_contexts", [1], "contexts must be strings"),
        ("context_chunk_ids", [1], "context IDs must be nonempty strings"),
        ("context_citation_ids", ["1"], "citation IDs must be positive integers"),
        ("context_citation_ids", [True], "citation IDs must be positive integers"),
        ("context_token_count", True, "nonnegative integer"),
        ("context_truncated", 0, "must be boolean"),
        ("context_corpus_version", True, "nonnegative integer"),
        ("citations", {}, "citations must be a list"),
        ("rendered_context", None, "rendered context must be a string"),
    ],
)
def test_strict_sample_validation_rejects_type_coercion(
    field: str,
    value: object,
    message: str,
) -> None:
    runner = _load_runner()
    sample = {**_valid_sample(), field: value}
    entry = {"id": "q1", "query": "question", "expected_answer": "reference"}

    with pytest.raises(ValueError, match=message):
        runner.validate_samples([sample], [entry], expected_corpus_version=7)


def test_strict_sample_validation_rejects_duplicate_and_unaligned_ids() -> None:
    runner = _load_runner()
    entry = {"id": "q1", "query": "question", "expected_answer": "reference"}

    duplicate_chunks = {
        **_valid_sample(),
        "retrieved_contexts": ["one", "two"],
        "context_chunk_ids": ["same", "same"],
        "context_citation_ids": [1, 2],
    }
    with pytest.raises(ValueError, match="context IDs must be unique"):
        runner.validate_samples([duplicate_chunks], [entry])

    duplicate_citations = {
        **_valid_sample(),
        "retrieved_contexts": ["one", "two"],
        "context_chunk_ids": ["one", "two"],
        "context_citation_ids": [1, 1],
    }
    with pytest.raises(ValueError, match="citation IDs must be unique"):
        runner.validate_samples([duplicate_citations], [entry])

    unaligned = {**_valid_sample(), "context_citation_ids": []}
    with pytest.raises(ValueError, match="not aligned"):
        runner.validate_samples([unaligned], [entry])

    duplicate_answer_citations = {
        **_valid_sample(),
        "citations": [{"citation_id": 1}, {"citation_id": 1}],
    }
    with pytest.raises(ValueError, match="cited IDs must be unique"):
        runner.validate_samples([duplicate_answer_citations], [entry])

    entries = [entry, {**entry, "id": "q2"}]
    samples = [_valid_sample(), _valid_sample()]
    with pytest.raises(ValueError, match="sample IDs must be unique"):
        runner.validate_samples(samples, entries)


def test_deterministic_evaluator_requires_explicit_context_citation_ids() -> None:
    runner = _load_runner()
    sample = _valid_sample()
    sample.pop("context_citation_ids")

    with pytest.raises(ValueError, match="missing: context_citation_ids"):
        runner.evaluate_deterministic_samples([sample])


def test_prefix_replay_validates_full_bundle_before_selecting_samples(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    source_samples = [
        _valid_sample("q1"),
        {
            **_valid_sample("q2"),
            "generation_prompt_sha256": "b" * 64,
        },
    ]
    generation = runner.sample_generation_provenance(source_samples)
    results_path, _ = runner.write_evaluation_artifacts(
        output_dir=tmp_path,
        label="full",
        kind="rag-samples",
        results=source_samples,
        metadata={
            "dataset_sha256": "dataset-hash",
            "sample_count": 2,
            "mode": "hybrid",
            "collection_corpus_version": 7,
            **generation,
        },
        metrics={"sample_count": 2},
    )

    evaluated, producer, source_count = runner.load_replay_bundle(
        samples_path=results_path,
        dataset_sha256="dataset-hash",
        configuration={"mode": "hybrid"},
        limit=1,
        expected_answer_model=None,
    )

    assert source_count == 2
    assert producer["sample_count"] == 2
    assert [sample["id"] for sample in evaluated] == ["q1"]
    assert runner.evaluation_scope_provenance(
        evaluated,
        source_sample_count=source_count,
    ) == {
        "source_sample_count": 2,
        "evaluated_sample_count": 1,
        "evaluated_sample_ids": ["q1"],
    }


def test_collection_rejects_corpus_version_change(tmp_path: Path) -> None:
    runner = _load_runner()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sources": [
                    {"source_id": "source.pdf", "source_sha256": "f" * 64}
                ]
            }
        ),
        encoding="utf-8",
    )
    entries = [
        {
            "id": "q1",
            "query": "question",
            "expected_answer": "reference",
            "relevant_passages": [{"source_id": "source.pdf"}],
        }
    ]

    with pytest.raises(RuntimeError, match="corpus version changed"):
        runner.collect_samples(
            client=FakeClient(corpus_versions=(7, 8)),
            entries=entries,
            mode="hybrid",
            candidate_k=20,
            max_chunks=8,
            max_context_tokens=1200,
            vector_weight=0.7,
            rerank=True,
            rerank_weight=None,
            manifest_path=manifest,
        )


def test_evaluates_deterministic_proxies_without_ragas() -> None:
    runner = _load_runner()
    results = runner.evaluate_deterministic_samples(
        [
            {
                "id": "q1",
                "response": "supported answer [1]",
                "reference": "supported answer",
                "retrieved_contexts": ["supported answer"],
                "citations": [{"citation_id": 1}],
                "context_citation_ids": [1],
            }
        ]
    )

    assert results[0]["deterministic"][
        "deterministic_reference_token_f1"
    ] == 1.0
    assert "ragas" not in results[0]


def test_replay_provenance_rejects_tampered_samples(tmp_path: Path) -> None:
    runner = _load_runner()
    samples = tmp_path / "run.rag-samples.results.jsonl"
    samples.write_text('{"id":"q1"}\n', encoding="utf-8", newline="\n")
    summary = tmp_path / "run.rag-samples.summary.json"
    summary.write_text(
        json.dumps(
            {
                "metadata": {
                    "results_sha256": runner.sha256_file(samples),
                    "dataset_sha256": "dataset-hash",
                    "sample_count": 1,
                    "mode": "hybrid",
                    "candidate_k": 20,
                    "max_chunks": 8,
                    "max_context_tokens": 1200,
                    "vector_weight": 0.7,
                    "rerank": True,
                    "rerank_weight": None,
                },
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )
    configuration = {
        "mode": "hybrid",
        "candidate_k": 20,
        "max_chunks": 8,
        "max_context_tokens": 1200,
        "vector_weight": 0.7,
        "rerank": True,
        "rerank_weight": None,
    }

    metadata = runner.load_and_validate_sample_provenance(
        samples_path=samples,
        dataset_sha256="dataset-hash",
        configuration=configuration,
        sample_count=1,
    )
    assert metadata["source_samples_sha256"] == runner.sha256_file(samples)

    samples.write_text('{"id":"changed"}\n', encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="hash does not match"):
        runner.load_and_validate_sample_provenance(
            samples_path=samples,
            dataset_sha256="dataset-hash",
            configuration=configuration,
            sample_count=1,
        )


def test_replay_provenance_rejects_configuration_mismatch(tmp_path: Path) -> None:
    runner = _load_runner()
    samples = tmp_path / "run.rag-samples.results.jsonl"
    samples.write_text('{"id":"q1"}\n', encoding="utf-8", newline="\n")
    summary = tmp_path / "run.rag-samples.summary.json"
    summary.write_text(
        json.dumps(
            {
                "metadata": {
                    "results_sha256": runner.sha256_file(samples),
                    "dataset_sha256": "dataset-hash",
                    "sample_count": 1,
                    "mode": "vector",
                },
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="configuration differs"):
        runner.load_and_validate_sample_provenance(
            samples_path=samples,
            dataset_sha256="dataset-hash",
            configuration={"mode": "hybrid"},
            sample_count=1,
        )

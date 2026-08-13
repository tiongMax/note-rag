from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from note_rag.context import ContextPackage


def _load_runner() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "run_guardrail_benchmark.py"
    spec = importlib.util.spec_from_file_location("guardrail_runner", path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    return runner


class FakeGuardrailService:
    def __init__(self) -> None:
        self.input_calls = 0
        self.context_calls = 0
        self.output_calls = 0

    @staticmethod
    def _decision(stage: str, unsafe: bool) -> SimpleNamespace:
        action = "block" if unsafe else "allow"
        if stage == "context" and unsafe:
            action = "filter"
        return SimpleNamespace(
            stage=stage,
            action=action,
            reason_codes=("synthetic_attack_marker",) if unsafe else (),
            latency_ms=0.01,
        )

    def inspect_input(self, question: str) -> SimpleNamespace:
        self.input_calls += 1
        return self._decision("input", "attack_marker" in question)

    def sanitize_context(self, context: ContextPackage) -> SimpleNamespace:
        self.context_calls += 1
        unsafe = any("attack_marker" in chunk.text for chunk in context.chunks)
        return SimpleNamespace(
            context=context,
            filtered_chunk_count=int(unsafe),
            decision=self._decision("context", unsafe),
        )

    def validate_output(
        self,
        *,
        question: str,
        answer: str,
        context: ContextPackage,
    ) -> SimpleNamespace:
        del question, context
        self.output_calls += 1
        unsafe = "attack_marker" in answer
        return SimpleNamespace(
            answer=answer,
            decision=self._decision("output", unsafe),
            lexical_groundedness_proxy=0.0 if unsafe else 1.0,
            lexical_relevance_proxy=0.0 if unsafe else 1.0,
        )


def _case(
    runner: ModuleType,
    identifier: str,
    stage: str,
    *,
    expected_unsafe: bool,
) -> Any:
    marker = "attack_marker" if expected_unsafe else "ordinary material"
    return runner.GuardrailCase(
        schema_version="1.0",
        ruleset_version="1.0",
        id=identifier,
        stage=stage,
        label="unsafe" if expected_unsafe else "safe",
        category="synthetic_test",
        expected_unsafe=expected_unsafe,
        hard_negative=False,
        question=marker if stage == "input" else "ordinary question",
        contexts=(marker if stage == "context" else "ordinary support",),
        context_roles=("supporting",),
        answer=marker if stage == "output" else "ordinary answer",
        rationale="synthetic test case",
        tags=("test",),
    )


def test_evaluates_each_service_stage_and_builds_confusion_matrices() -> None:
    runner = _load_runner()
    service = FakeGuardrailService()
    cases = [
        _case(
            runner,
            f"{stage}-unsafe",
            stage,
            expected_unsafe=True,
        )
        for stage in runner.STAGES
    ] + [
        _case(
            runner,
            f"{stage}-safe",
            stage,
            expected_unsafe=False,
        )
        for stage in runner.STAGES
    ]

    rows = runner.evaluate_cases(service, cases)
    summary = runner.summarize_classification(rows)

    assert service.input_calls == 2
    assert service.context_calls == 2
    assert service.output_calls == 2
    assert summary["overall"]["confusion_matrix"] == {
        "true_positive": 3,
        "false_positive": 0,
        "true_negative": 3,
        "false_negative": 0,
    }
    for stage in runner.STAGES:
        assert summary["by_stage"][stage]["confusion_matrix"] == {
            "true_positive": 1,
            "false_positive": 0,
            "true_negative": 1,
            "false_negative": 0,
        }
    assert "not a semantic safety evaluation" in summary["metric_scope"]


def test_reports_false_positive_and_false_negative_case_ids_by_stage() -> None:
    runner = _load_runner()
    rows = [
        {
            "id": "input-fp",
            "stage": "input",
            "expected_unsafe": False,
            "predicted_unsafe": True,
            "hard_negative": True,
        },
        {
            "id": "input-fn",
            "stage": "input",
            "expected_unsafe": True,
            "predicted_unsafe": False,
            "hard_negative": False,
        },
        {
            "id": "context-tn",
            "stage": "context",
            "expected_unsafe": False,
            "predicted_unsafe": False,
            "hard_negative": False,
        },
        {
            "id": "output-tp",
            "stage": "output",
            "expected_unsafe": True,
            "predicted_unsafe": True,
            "hard_negative": False,
        },
    ]

    summary = runner.summarize_classification(rows)

    assert summary["by_stage"]["input"]["false_positive_ids"] == ["input-fp"]
    assert summary["by_stage"]["input"]["false_negative_ids"] == ["input-fn"]
    assert summary["hard_negative_false_positive_ids"] == ["input-fp"]


def test_rejects_a_decision_at_the_wrong_stage() -> None:
    runner = _load_runner()
    service = FakeGuardrailService()
    service.inspect_input = lambda _question: service._decision("output", False)  # type: ignore[method-assign]
    case = _case(runner, "wrong-stage", "input", expected_unsafe=False)

    with pytest.raises(ValueError, match="returned stage 'output'"):
        runner.evaluate_case(service, case)


def test_benchmarks_complete_path_after_warmup_with_nanosecond_clock() -> None:
    runner = _load_runner()
    service = FakeGuardrailService()
    fixture = runner.build_benchmark_fixtures()["normal"]

    result = runner.benchmark_fixture(
        service,
        fixture,
        iterations=7,
        warmup_iterations=3,
    )

    assert service.input_calls == 10
    assert service.context_calls == 10
    assert service.output_calls == 10
    assert result["clock"] == "time.perf_counter_ns"
    assert result["complete_path_stages"] == ["input", "context", "output"]
    assert result["latency"]["count"] == 7
    assert result["latency"]["median_ns"] >= 0
    assert result["latency"]["p95_ns"] >= result["latency"]["median_ns"]
    assert result["latency"]["p99_ns"] >= result["latency"]["p95_ns"]


def test_normal_and_max_fixtures_pin_declared_token_loads() -> None:
    runner = _load_runner()

    fixtures = runner.build_benchmark_fixtures()

    assert set(fixtures) == {"normal", "max"}
    max_fixture = fixtures["max"]
    assert max_fixture.expected_question_tokens == 512
    assert max_fixture.expected_context_tokens == 1200
    assert len(max_fixture.context.chunks) == 8
    assert max_fixture.expected_answer_tokens == 1024
    assert runner._token_count(max_fixture.question) == 512
    assert sum(
        runner._token_count(chunk.text) for chunk in max_fixture.context.chunks
    ) == (1200)
    assert runner._token_count(max_fixture.answer) == 1024
    assert fixtures["normal"].expected_question_tokens < 512
    assert fixtures["normal"].expected_context_tokens < 1200
    assert fixtures["normal"].expected_answer_tokens < 1024


def test_cli_defaults_to_at_least_ten_thousand_measured_iterations() -> None:
    runner = _load_runner()

    args = runner._parser().parse_args([])

    assert args.iterations >= 10_000
    assert args.warmup_iterations > 0
    assert args.fixture == "all"


def test_latency_summary_reports_median_p95_and_p99() -> None:
    runner = _load_runner()

    summary = runner.summarize_latency_ns([100, 200, 300, 400, 500])

    assert summary["median_ns"] == 300
    assert summary["p95_ns"] == pytest.approx(480)
    assert summary["p99_ns"] == pytest.approx(496)
    assert summary["median_ms"] == pytest.approx(0.0003)


def test_checked_in_corpus_is_hashed_versioned_synthetic_regression() -> None:
    runner = _load_runner()
    root = Path(__file__).parents[2]
    cases_path = root / "benchmarks" / "guardrails" / "cases.jsonl"
    manifest_path = root / "benchmarks" / "guardrails" / "manifest.json"

    cases = runner.load_cases(cases_path)
    manifest = runner.validate_manifest(
        manifest_path,
        cases_path=cases_path,
        cases=cases,
    )

    assert manifest["synthetic"] is True
    assert manifest["evaluation_declaration"]["held_out"] is False
    assert manifest["counts"]["overall"] == len(cases)
    assert set(manifest["counts"]["by_stage"]) == set(runner.STAGES)
    assert any(case.hard_negative for case in cases)
    for stage in runner.STAGES:
        stage_cases = [case for case in cases if case.stage == stage]
        assert {case.expected_unsafe for case in stage_cases} == {False, True}
        assert any(case.hard_negative for case in stage_cases)


def test_manifest_validation_rejects_tampered_case_corpus(tmp_path: Path) -> None:
    runner = _load_runner()
    cases_path = tmp_path / "cases.jsonl"
    manifest_path = tmp_path / "manifest.json"
    records = []
    for stage in runner.STAGES:
        for unsafe in (False, True):
            records.append(
                {
                    "schema_version": "1.0",
                    "ruleset_version": "1.0",
                    "id": f"{stage}-{unsafe}",
                    "stage": stage,
                    "label": "unsafe" if unsafe else "safe",
                    "category": "synthetic_test",
                    "expected_unsafe": unsafe,
                    "hard_negative": False,
                    "question": "question",
                    "contexts": ["context"],
                    "context_roles": ["supporting"],
                    "answer": "answer",
                    "rationale": "test",
                    "tags": ["test"],
                }
            )
    cases_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    cases = runner.load_cases(cases_path)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "ruleset_version": "1.0",
                "synthetic": True,
                "split": "regression",
                "evaluation_declaration": {"held_out": False},
                "cases_sha256": "0" * 64,
                "counts": {
                    "overall": len(cases),
                    "by_stage": {stage: 2 for stage in runner.STAGES},
                    "by_label": {"safe": 3, "unsafe": 3},
                    "by_stage_and_label": {
                        stage: {"safe": 1, "unsafe": 1} for stage in runner.STAGES
                    },
                    "hard_negatives": 0,
                },
                "allowed_values": {
                    "stage": list(runner.STAGES),
                    "label": ["safe", "unsafe"],
                    "context_role": [
                        "supporting",
                        "distractor",
                        "adversarial",
                    ],
                },
                "cases": "benchmarks/guardrails/cases.jsonl",
                "dataset": "test",
                "dataset_version": "1.0.0",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hash does not match"):
        runner.validate_manifest(
            manifest_path,
            cases_path=cases_path,
            cases=cases,
        )


def test_report_records_machine_python_and_corpus_hashes(tmp_path: Path) -> None:
    runner = _load_runner()
    cases_path = tmp_path / "cases.jsonl"
    manifest_path = tmp_path / "manifest.json"
    cases_path.write_text("{}\n", encoding="utf-8")
    manifest_path.write_text("{}\n", encoding="utf-8")
    rows = [
        {
            "id": "input-safe",
            "stage": "input",
            "expected_unsafe": False,
            "predicted_unsafe": False,
            "hard_negative": False,
        },
        {
            "id": "context-safe",
            "stage": "context",
            "expected_unsafe": False,
            "predicted_unsafe": False,
            "hard_negative": False,
        },
        {
            "id": "output-safe",
            "stage": "output",
            "expected_unsafe": False,
            "predicted_unsafe": False,
            "hard_negative": False,
        },
    ]

    report = runner.build_report(
        label="smoke",
        cases_path=cases_path,
        manifest_path=manifest_path,
        manifest={
            "synthetic": True,
            "evaluation_declaration": {"held_out": False},
        },
        ruleset_version="1.0",
        rows=rows,
        benchmarks={},
    )

    assert len(report["runtime"]["machine_sha256"]) == 64
    assert len(report["runtime"]["python_sha256"]) == 64
    assert len(report["implementation"]["combined_source_sha256"]) == 64
    assert len(report["implementation"]["source_sha256"]) == 4
    assert "git_dirty" in report["implementation"]
    assert len(report["corpus_hashes"]["cases_sha256"]) == 64
    assert len(report["corpus_hashes"]["manifest_sha256"]) == 64
    assert len(report["corpus_hashes"]["combined_sha256"]) == 64
    assert "must not be described as semantic" in report["metric_scope"]


def test_writes_per_case_and_summary_artifacts_with_results_hash(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    rows = [{"id": "case-1", "stage": "input"}]

    results_path, summary_path = runner.write_report(
        tmp_path,
        label="smoke",
        rows=rows,
        report={"schema_version": "1.0"},
    )

    assert json.loads(results_path.read_text(encoding="utf-8"))["id"] == ("case-1")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["results_sha256"] == runner.sha256_file(results_path)

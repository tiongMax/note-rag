from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_conversation_memory_benchmark.py"


def load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("memory_benchmark", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ExactTopicProvider:
    model_name = "exact-topic-test"
    dimension = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vector(query)

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.casefold()
        return [float("orion" in lowered), float("atlas" in lowered)]


def test_loads_complete_hash_bound_dataset() -> None:
    runner = load_runner()

    cases = runner.load_cases(runner.DEFAULT_CASES, runner.DEFAULT_MANIFEST)

    assert len(cases) == 50
    assert cases[0].identifier == "memory-001"
    assert cases[-1].identifier == "memory-050"


def test_rejects_dataset_hash_mismatch(tmp_path: Path) -> None:
    runner = load_runner()
    cases_path = tmp_path / "cases.jsonl"
    manifest_path = tmp_path / "manifest.json"
    cases_path.write_text(
        '{"schema_version":"1.0","id":"x","subject":"s",'
        '"fact":"f","query":"q","expected_answer":"a"}\n',
        encoding="utf-8",
    )
    manifest_path.write_text(
        '{"schema_version":"1.0","held_out":false,"split":"regression",'
        '"cases_sha256":"wrong","case_count":1}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hash"):
        runner.load_cases(cases_path, manifest_path)


def test_evaluation_reports_fact_retention_and_token_reduction() -> None:
    runner = load_runner()
    case = runner.Case(
        identifier="memory-test",
        subject="Orion deployment",
        fact="The Orion code is amber.",
        query="What was the Orion code?",
        expected_answer="amber",
    )

    records, metrics = runner.evaluate(
        [case],
        ExactTopicProvider(),
        history_budget=2000,
        recent_turns=1,
        semantic_k=1,
        minimum_similarity=0.0,
        summary_max_tokens=64,
    )

    assert metrics["full_history_fact_retention_accuracy"] == 1.0
    assert metrics["recent_only_fact_retention_accuracy"] == 0.0
    assert metrics["memory_fact_retention_accuracy"] == 1.0
    assert records[0]["memory_history_tokens"] < records[0]["full_history_tokens"]


def test_previous_recent_only_selection_stops_at_first_oversized_message() -> None:
    runner = load_runner()
    messages = [
        (0, "user", "old", 1),
        (1, "assistant", "too large", 10),
        (2, "user", "recent", 1),
    ]

    selected = runner.select_previous_recent_only_history(
        messages,
        maximum_messages=20,
        maximum_tokens=5,
    )

    assert selected == [(2, "user", "recent", 1)]


def test_git_provenance_distinguishes_clean_output_from_failure(
    monkeypatch,
) -> None:
    runner = load_runner()

    class Result:
        stdout = ""

    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: Result())

    assert runner.git_value("status", "--porcelain", allow_empty=True) == ""
    assert runner.git_value("status", "--porcelain") is None

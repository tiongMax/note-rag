import json

import pytest

from note_rag.evaluation.dataset import load_benchmark


def _case(case_id: str = "q1") -> dict:
    return {
        "id": case_id,
        "question": "What happened?",
        "answerable": True,
        "reference_answer": "A release happened.",
        "evidence": [
            {
                "document": "notes.md",
                "quote": "A release happened.",
                "group": "release",
                "relevance": 3,
            }
        ],
    }


def test_load_benchmark_validates_and_skips_comments(tmp_path):
    path = tmp_path / "benchmark.jsonl"
    path.write_text(
        "# benchmark v1\n" + json.dumps(_case()) + "\n",
        encoding="utf-8",
    )

    cases = load_benchmark(path)

    assert [case.id for case in cases] == ["q1"]
    assert cases[0].expected_behavior == "answer"


def test_load_benchmark_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "benchmark.jsonl"
    row = json.dumps(_case())
    path.write_text(f"{row}\n{row}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate case id"):
        load_benchmark(path)


def test_unanswerable_case_cannot_contain_evidence(tmp_path):
    payload = _case()
    payload.update(
        answerable=False,
        reference_answer=None,
        expected_behavior="refuse",
    )
    path = tmp_path / "benchmark.jsonl"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot contain evidence"):
        load_benchmark(path)

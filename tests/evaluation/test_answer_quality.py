from __future__ import annotations

import pytest

from note_rag.evaluation import evaluate_deterministic_answer_quality


def test_scores_lexical_overlap_and_citations_as_deterministic_proxies() -> None:
    scores = evaluate_deterministic_answer_quality(
        response="Alpha beta beta extra [1] [3] [3]",
        reference="alpha beta beta gamma",
        retrieved_contexts=["Alpha beta support", "unrelated context"],
    )

    assert scores == {
        "deterministic_reference_token_precision": pytest.approx(3 / 4),
        "deterministic_reference_token_recall": pytest.approx(3 / 4),
        "deterministic_reference_token_f1": pytest.approx(3 / 4),
        "deterministic_answer_support_by_context": pytest.approx(2 / 4),
        "deterministic_reference_coverage_by_context": pytest.approx(2 / 4),
        "deterministic_citation_presence": 1.0,
        "deterministic_valid_citation_rate": pytest.approx(1 / 2),
    }


def test_uses_multiset_overlap_without_ignoring_case() -> None:
    scores = evaluate_deterministic_answer_quality(
        response="TOKEN token token",
        reference="token token",
        retrieved_contexts=["Token"],
    )

    assert scores["deterministic_reference_token_precision"] == pytest.approx(
        2 / 3
    )
    assert scores["deterministic_reference_token_recall"] == 1.0
    assert scores["deterministic_answer_support_by_context"] == pytest.approx(
        1 / 3
    )


def test_accepts_explicit_available_citation_ids() -> None:
    scores = evaluate_deterministic_answer_quality(
        response="Supported statement [4] [9]",
        reference="supported statement",
        retrieved_contexts=["supported statement"],
        available_citation_ids={4},
    )

    assert scores["deterministic_citation_presence"] == 1.0
    assert scores["deterministic_valid_citation_rate"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("response", "reference", "contexts", "citation_presence"),
    [
        ("", "", [], 0.0),
        ("answer", "", [], 0.0),
        ("", "reference", ["reference"], 0.0),
        ("[1]", "", [], 1.0),
    ],
)
def test_handles_empty_values_deterministically(
    response: str,
    reference: str,
    contexts: list[str],
    citation_presence: float,
) -> None:
    scores = evaluate_deterministic_answer_quality(
        response=response,
        reference=reference,
        retrieved_contexts=contexts,
    )

    assert scores["deterministic_citation_presence"] == citation_presence
    assert scores["deterministic_valid_citation_rate"] == 0.0
    assert all(0.0 <= score <= 1.0 for score in scores.values())
    if not response or not reference:
        assert scores["deterministic_reference_token_f1"] == 0.0


def test_metric_names_cannot_be_mistaken_for_ragas_metrics() -> None:
    scores = evaluate_deterministic_answer_quality(
        response="answer",
        reference="answer",
        retrieved_contexts=["answer"],
    )

    assert scores
    assert all(name.startswith("deterministic_") for name in scores)
    assert all("ragas" not in name for name in scores)

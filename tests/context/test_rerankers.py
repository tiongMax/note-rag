import pytest

from note_rag.context import LexicalReranker


def test_lexical_reranker_scores_query_overlap() -> None:
    scores = LexicalReranker().score(
        "apple nutrition",
        ["rock formations", "apple nutrition guide", "apple orchard"],
    )

    assert scores[1] > scores[2] > scores[0]
    assert all(0.0 <= score <= 1.0 for score in scores)


@pytest.mark.parametrize("documents", [[], [""]])
def test_lexical_reranker_handles_empty_input(documents: list[str]) -> None:
    assert LexicalReranker().score("", documents) == [0.0] * len(documents)

import pytest

from note_rag.context import CrossEncoderReranker, LexicalReranker


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


class FakeCrossEncoder:
    def predict(
        self,
        pairs: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> list[float]:
        assert batch_size == 8
        assert show_progress_bar is False
        return [0.9 if "relevant" in document else 0.1 for _, document in pairs]


def test_cross_encoder_is_lazy_and_scores_query_document_pairs() -> None:
    created = []

    def factory(model_name: str, **kwargs: str) -> FakeCrossEncoder:
        created.append((model_name, kwargs))
        return FakeCrossEncoder()

    reranker = CrossEncoderReranker(
        "local-test-model",
        device="cpu",
        batch_size=8,
        model_factory=factory,
    )

    assert created == []
    assert reranker.score("query", ["noise", "relevant passage"]) == [
        0.1,
        0.9,
    ]
    assert created == [("local-test-model", {"device": "cpu"})]


def test_cross_encoder_rejects_unbounded_logits() -> None:
    class LogitModel:
        def predict(self, *args: object, **kwargs: object) -> list[float]:
            return [8.2]

    reranker = CrossEncoderReranker(
        model_factory=lambda *args, **kwargs: LogitModel()
    )

    with pytest.raises(ValueError, match="sigmoid"):
        reranker.score("query", ["document"])

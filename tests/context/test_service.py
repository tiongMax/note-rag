import uuid

import pytest

from note_rag.context import ContextBuilder, LexicalReranker
from note_rag.retrieval import (
    RetrievalHit,
    RetrievalResult,
    SearchMode,
)


class StubRetriever:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits

    def search(self, query: str, **_kwargs) -> RetrievalResult:
        return RetrievalResult(
            query=query.strip(),
            mode=SearchMode.HYBRID,
            hits=self.hits,
        )


def hit(
    text: str,
    score: float,
    *,
    filename: str = "notes.txt",
    metadata: dict[str, object] | None = None,
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename=filename,
        media_type="text/plain",
        position=0,
        text=text,
        token_count=len(text.split()),
        token_start=0,
        token_end=len(text.split()),
        char_start=0,
        char_end=len(text),
        source_metadata=metadata or {},
        score=score,
    )


def test_reranks_then_removes_duplicate_text() -> None:
    candidates = [
        hit("unrelated geology", 0.9),
        hit("apple banana nutrition", 0.5, metadata={"page": 2}),
        hit("  APPLE   banana nutrition  ", 0.4),
    ]
    builder = ContextBuilder(
        StubRetriever(candidates),
        LexicalReranker(),
    )

    result = builder.build(
        "apple banana",
        rerank_weight=1.0,
        max_context_tokens=200,
    )

    assert result.chunks[0].text == "apple banana nutrition"
    assert result.chunks[0].rerank_score is not None
    assert result.chunks[0].source_metadata == {"page": 2}
    assert result.duplicates_removed == 1
    assert [chunk.citation_id for chunk in result.chunks] == [1, 2]


def test_context_respects_budget_and_truncates_at_token_boundary() -> None:
    candidate = hit(
        "one two three four five six seven eight nine ten " * 4,
        1.0,
        filename="long.txt",
        metadata={"page": 7, "section": "budget"},
    )
    builder = ContextBuilder(StubRetriever([candidate]), LexicalReranker())

    result = builder.build(
        "numbers",
        max_context_tokens=40,
        max_chunks=1,
    )

    assert result.token_count <= 40
    assert result.chunks[0].truncated is True
    assert result.chunks[0].text == "one two three four"
    assert result.chunks[0].source_metadata["page"] == 7
    assert result.truncated is True
    assert "[1] Source: long.txt" in result.context


def test_reranking_can_be_disabled() -> None:
    class FailingReranker:
        model_name = "must-not-run"

        def score(self, query: str, documents: list[str]) -> list[float]:
            raise AssertionError("reranker should not run")

    candidates = [hit("first", 0.8), hit("second", 0.2)]
    result = ContextBuilder(
        StubRetriever(candidates),
        FailingReranker(),
    ).build("query", rerank=False, max_context_tokens=100)

    assert [chunk.text for chunk in result.chunks] == ["first", "second"]
    assert all(chunk.rerank_score is None for chunk in result.chunks)
    assert result.reranker_model is None


def test_rejects_invalid_reranker_output() -> None:
    class BrokenReranker:
        model_name = "broken"

        def score(self, query: str, documents: list[str]) -> list[float]:
            return []

    with pytest.raises(ValueError, match="wrong score count"):
        ContextBuilder(
            StubRetriever([hit("candidate", 1.0)]),
            BrokenReranker(),
        ).build("query")


def test_rejects_out_of_range_reranker_score() -> None:
    class BrokenReranker:
        model_name = "broken"

        def score(self, query: str, documents: list[str]) -> list[float]:
            return [1.1]

    with pytest.raises(ValueError, match="between zero and one"):
        ContextBuilder(
            StubRetriever([hit("candidate", 1.0)]),
            BrokenReranker(),
        ).build("query")

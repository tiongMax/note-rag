import pytest

from note_rag.chunking import TokenChunker


def test_chunks_by_token_budget_with_overlap_and_metadata() -> None:
    text = "zero one two three four five six seven"

    chunks = TokenChunker(chunk_size=4, chunk_overlap=1).chunk(
        text,
        source_id="document-1",
    )

    assert [chunk.text for chunk in chunks] == [
        "zero one two three",
        "three four five six",
        "six seven",
    ]
    assert [chunk.metadata.token_count for chunk in chunks] == [4, 4, 2]
    assert [chunk.metadata.token_start for chunk in chunks] == [0, 3, 6]
    assert [chunk.metadata.token_end for chunk in chunks] == [4, 7, 8]
    assert all(chunk.metadata.source_id == "document-1" for chunk in chunks)
    assert text[chunks[1].metadata.char_start : chunks[1].metadata.char_end] == (
        chunks[1].text
    )


def test_empty_input_has_no_chunks() -> None:
    assert TokenChunker(chunk_size=10, chunk_overlap=2).chunk(" \n") == []


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (5, -1), (5, 5), (5, 6)],
)
def test_rejects_invalid_windows(chunk_size: int, chunk_overlap: int) -> None:
    with pytest.raises(ValueError):
        TokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

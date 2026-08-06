import pytest

from note_rag.chunking import RecursiveChunker


def test_preserves_heading_sections_with_bounded_overlap() -> None:
    text = "# Alpha\nalpha one two\n\n# Beta\nbeta three four"

    chunks = RecursiveChunker(chunk_size=7, chunk_overlap=2).chunk(
        text,
        source_id="structured.md",
    )

    beta_start = text.index("# Beta")
    beta_end = len(text)
    beta_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata.char_start <= beta_start
        and chunk.metadata.char_end >= beta_end
    ]

    assert beta_chunks
    assert all(chunk.metadata.token_count <= 7 for chunk in chunks)
    assert chunks[1].metadata.token_start == (
        chunks[0].metadata.token_end - 2
    )
    assert all(
        chunk.metadata.source_id == "structured.md" for chunk in chunks
    )


def test_uses_token_fallback_for_oversized_unstructured_text() -> None:
    text = "zero one two three four five six seven eight nine"

    chunks = RecursiveChunker(chunk_size=4, chunk_overlap=1).chunk(text)

    assert all(chunk.metadata.token_count <= 4 for chunk in chunks)
    assert chunks[0].metadata.token_start == 0
    assert chunks[-1].metadata.token_end == 10
    assert all(
        current.metadata.token_end - following.metadata.token_start == 1
        for current, following in zip(chunks, chunks[1:])
    )


def test_source_ranges_reconstruct_chunk_text() -> None:
    text = "First paragraph has words.\n\nSecond paragraph has other words."

    chunks = RecursiveChunker(chunk_size=8, chunk_overlap=2).chunk(text)

    assert all(
        text[chunk.metadata.char_start : chunk.metadata.char_end] == chunk.text
        for chunk in chunks
    )


def test_empty_input_has_no_recursive_chunks() -> None:
    assert RecursiveChunker(chunk_size=10, chunk_overlap=2).chunk(" \n") == []


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (5, -1), (5, 5), (5, 6)],
)
def test_rejects_invalid_recursive_windows(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    with pytest.raises(ValueError):
        RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

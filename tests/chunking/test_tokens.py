from note_rag.chunking import RegexTokenCounter


def test_counts_words_numbers_and_punctuation_deterministically() -> None:
    counter = RegexTokenCounter()

    tokens = counter.tokenize("RAG is useful—really useful!")

    assert [token.text for token in tokens] == [
        "RAG",
        "is",
        "useful",
        "—",
        "really",
        "useful",
        "!",
    ]
    assert counter.count("RAG is useful—really useful!") == 7


def test_whitespace_has_no_tokens() -> None:
    assert RegexTokenCounter().count(" \n\t") == 0

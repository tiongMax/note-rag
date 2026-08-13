from note_rag.chunking import lexical_term_frequencies, lexical_terms


def test_lexical_terms_normalize_and_bound_index_keys() -> None:
    oversized = "A" * 200

    terms = lexical_terms(f"Apple apple snake_case {oversized}!")

    assert terms[:3] == ["apple", "apple", "snake_case"]
    assert terms[3] == "a" * 128


def test_lexical_term_frequencies_preserve_repetition() -> None:
    assert lexical_term_frequencies("Apple apple pear") == {
        "apple": 2,
        "pear": 1,
    }

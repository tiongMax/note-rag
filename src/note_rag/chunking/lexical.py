"""Deterministic lexical token statistics used by PostgreSQL BM25."""

from __future__ import annotations

import re
from collections import Counter

_LEXICAL_TERM = re.compile(r"\w+", re.UNICODE)
_MAX_TERM_LENGTH = 128


def lexical_terms(text: str) -> list[str]:
    """Return normalized word-like terms, bounded for indexed storage."""

    return [
        match.group(0).casefold()[:_MAX_TERM_LENGTH]
        for match in _LEXICAL_TERM.finditer(text)
    ]


def lexical_term_frequencies(text: str) -> Counter[str]:
    return Counter(lexical_terms(text))

"""A lightweight deterministic tokenizer used for chunk budgets."""

import re
from dataclasses import dataclass

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class Token:
    """One token and its half-open character range in the source text."""

    text: str
    start: int
    end: int


class RegexTokenCounter:
    """Count word-like units and punctuation without model dependencies.

    This is deliberately a budget tokenizer, not an embedding-model tokenizer.
    A later embedding provider can replace it while keeping the chunker contract.
    """

    def tokenize(self, text: str) -> tuple[Token, ...]:
        return tuple(
            Token(text=match.group(), start=match.start(), end=match.end())
            for match in TOKEN_PATTERN.finditer(text)
        )

    def count(self, text: str) -> int:
        return len(self.tokenize(text))

"""Reranker contracts and a deterministic local baseline."""

import math
from typing import Protocol

from note_rag.chunking import RegexTokenCounter


class Reranker(Protocol):
    @property
    def model_name(self) -> str: ...

    def score(self, query: str, documents: list[str]) -> list[float]: ...


class LexicalReranker:
    """Score query/document token overlap without an external model."""

    model_name = "lexical-overlap-v1"

    def __init__(self, token_counter: RegexTokenCounter | None = None) -> None:
        self.token_counter = token_counter or RegexTokenCounter()

    def score(self, query: str, documents: list[str]) -> list[float]:
        query_terms = self._terms(query)
        if not query_terms:
            return [0.0] * len(documents)
        scores = []
        for document in documents:
            document_terms = self._terms(document)
            if not document_terms:
                scores.append(0.0)
                continue
            overlap = len(query_terms & document_terms)
            scores.append(
                overlap / math.sqrt(len(query_terms) * len(document_terms))
            )
        return scores

    def _terms(self, text: str) -> set[str]:
        return {
            token.text.casefold()
            for token in self.token_counter.tokenize(text)
            if any(character.isalnum() for character in token.text)
        }

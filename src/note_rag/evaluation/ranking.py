"""Experimental lexical retrieval and rank-fusion helpers."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_TERM_PATTERN = re.compile(r"\w+", re.UNICODE)


def _terms(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TERM_PATTERN.finditer(text)]


@dataclass(frozen=True, slots=True)
class _Bm25Document:
    hit: Mapping[str, Any]
    term_frequencies: Counter[str]
    length: int


class Bm25Index:
    """Small, deterministic Okapi BM25 index for controlled experiments.

    This intentionally lives in the evaluation package rather than the request
    path. It lets the project measure whether BM25 is worth an operational
    dependency before adopting a PostgreSQL extension or search service.
    """

    def __init__(
        self,
        hits: Sequence[Mapping[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")
        self.k1 = k1
        self.b = b
        documents = []
        document_frequency: Counter[str] = Counter()
        for hit in hits:
            terms = _terms(str(hit["text"]))
            frequencies = Counter(terms)
            document_frequency.update(frequencies.keys())
            documents.append(_Bm25Document(hit, frequencies, len(terms)))
        self._documents = documents
        self._document_frequency = document_frequency
        self._average_length = (
            sum(document.length for document in documents) / len(documents)
            if documents
            else 0.0
        )

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def search(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        query_terms = set(_terms(query))
        if not query_terms or not self._documents:
            return []

        ranked: list[tuple[float, str, Mapping[str, Any]]] = []
        count = len(self._documents)
        for document in self._documents:
            score = 0.0
            length_ratio = (
                document.length / self._average_length
                if self._average_length
                else 0.0
            )
            for term in query_terms:
                frequency = document.term_frequencies.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequency[term]
                inverse_document_frequency = math.log(
                    1 + (count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length_ratio
                )
                score += inverse_document_frequency * (
                    frequency * (self.k1 + 1) / denominator
                )
            if score:
                hit_id = str(document.hit.get("chunk_id", document.hit.get("id")))
                ranked.append((score, hit_id, document.hit))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [
            {**hit, "score": score, "bm25_score": score}
            for score, _, hit in ranked[:limit]
        ]


def weighted_rrf(
    rankings: Sequence[tuple[Sequence[Mapping[str, Any]], float]],
    *,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse rankings by chunk ID and retain per-signal diagnostic scores."""

    if rrf_k < 0:
        raise ValueError("rrf_k cannot be negative")
    if not rankings:
        return []
    fused: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    for hits, weight in rankings:
        if weight < 0:
            raise ValueError("ranking weights cannot be negative")
        for rank, hit in enumerate(hits, start=1):
            hit_id = str(hit.get("chunk_id", hit.get("id")))
            fused.setdefault(hit_id, dict(hit))
            scores[hit_id] = scores.get(hit_id, 0.0) + weight / (rrf_k + rank)
            for score_name in ("vector_score", "keyword_score", "bm25_score"):
                if hit.get(score_name) is not None:
                    fused[hit_id][score_name] = hit[score_name]

    maximum = sum(weight for _, weight in rankings) / (rrf_k + 1)
    results = []
    for hit_id, hit in fused.items():
        hit["score"] = scores[hit_id] / maximum if maximum else 0.0
        results.append(hit)
    results.sort(
        key=lambda hit: (
            -float(hit["score"]),
            str(hit.get("chunk_id", hit.get("id"))),
        )
    )
    return results

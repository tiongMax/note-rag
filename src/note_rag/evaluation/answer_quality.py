"""Cheap deterministic proxies for offline answer-quality regression checks.

These metrics use lexical overlap and citation syntax. They are intentionally
named as deterministic proxies and must not be presented as semantic or RAGAS
scores.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Collection, Sequence

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
_CITATION_PATTERN = re.compile(r"\[(\d+)]")


def _tokens(text: str) -> list[str]:
    """Return normalized content tokens, excluding numeric citation markers."""

    without_citations = _CITATION_PATTERN.sub(" ", text)
    return [
        match.group(0).casefold()
        for match in _TOKEN_PATTERN.finditer(without_citations)
    ]


def _overlap_count(left: Sequence[str], right: Sequence[str]) -> int:
    """Count multiset token overlap so repeated words are not over-credited."""

    return sum((Counter(left) & Counter(right)).values())


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_deterministic_answer_quality(
    *,
    response: str,
    reference: str,
    retrieved_contexts: Sequence[str],
    available_citation_ids: Collection[int] | None = None,
) -> dict[str, float]:
    """Score one answer with deterministic lexical and citation proxies.

    Token overlap is case-insensitive and uses word-token multisets. Reference
    precision uses response tokens as its denominator, while reference recall
    uses reference tokens. Answer support and reference coverage measure the
    corresponding token overlap with all retrieved contexts combined.

    Citations are numeric markers such as ``[1]`` found in ``response``.
    Repeated markers count once. By default, valid IDs are the one-based
    positions of ``retrieved_contexts``; callers may pass the exact available
    IDs when they differ. Every metric is ``0.0`` when its denominator is
    empty, which keeps empty-input behavior explicit and deterministic.
    """

    response_tokens = _tokens(response)
    reference_tokens = _tokens(reference)
    context_tokens = [
        token
        for context in retrieved_contexts
        for token in _tokens(context)
    ]

    reference_overlap = _overlap_count(response_tokens, reference_tokens)
    reference_precision = _ratio(reference_overlap, len(response_tokens))
    reference_recall = _ratio(reference_overlap, len(reference_tokens))
    reference_f1 = (
        2 * reference_precision * reference_recall
        / (reference_precision + reference_recall)
        if reference_precision + reference_recall
        else 0.0
    )

    answer_context_overlap = _overlap_count(response_tokens, context_tokens)
    reference_context_overlap = _overlap_count(reference_tokens, context_tokens)

    cited_ids = {
        int(match.group(1)) for match in _CITATION_PATTERN.finditer(response)
    }
    valid_ids = (
        set(range(1, len(retrieved_contexts) + 1))
        if available_citation_ids is None
        else set(available_citation_ids)
    )
    valid_citation_count = len(cited_ids & valid_ids)

    return {
        "deterministic_reference_token_precision": reference_precision,
        "deterministic_reference_token_recall": reference_recall,
        "deterministic_reference_token_f1": reference_f1,
        "deterministic_answer_support_by_context": _ratio(
            answer_context_overlap,
            len(response_tokens),
        ),
        "deterministic_reference_coverage_by_context": _ratio(
            reference_context_overlap,
            len(reference_tokens),
        ),
        "deterministic_citation_presence": float(bool(cited_ids)),
        "deterministic_valid_citation_rate": _ratio(
            valid_citation_count,
            len(cited_ids),
        ),
    }

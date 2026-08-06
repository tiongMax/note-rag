"""Chunk-strategy-independent retrieval metrics."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GoldPassage:
    source_id: str
    char_start: int
    char_end: int

    def __post_init__(self) -> None:
        if self.char_start < 0 or self.char_end <= self.char_start:
            raise ValueError("gold passage must have a positive character range")

    @property
    def length(self) -> int:
        return self.char_end - self.char_start


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    source_id: str
    char_start: int
    char_end: int

    def __post_init__(self) -> None:
        if self.char_start < 0 or self.char_end <= self.char_start:
            raise ValueError("retrieved chunk must have a positive character range")


def passage_coverage(chunk: RetrievedChunk, passage: GoldPassage) -> float:
    """Return the proportion of a gold passage covered by one chunk."""

    if chunk.source_id != passage.source_id:
        return 0.0
    overlap = max(
        0,
        min(chunk.char_end, passage.char_end)
        - max(chunk.char_start, passage.char_start),
    )
    return overlap / passage.length


def _union_coverage(
    chunks: Sequence[RetrievedChunk],
    passage: GoldPassage,
) -> float:
    intervals = []
    for chunk in chunks:
        if chunk.source_id != passage.source_id:
            continue
        start = max(chunk.char_start, passage.char_start)
        end = min(chunk.char_end, passage.char_end)
        if start < end:
            intervals.append((start, end))
    if not intervals:
        return 0.0
    intervals.sort()
    covered = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            covered += current_end - current_start
            current_start, current_end = start, end
    covered += current_end - current_start
    return covered / passage.length


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def _ndcg(
    grades: Sequence[float],
    ideal_grades: Sequence[float] | None = None,
) -> float:
    if not grades or not any(grades):
        return 0.0

    def dcg(values: Sequence[float]) -> float:
        return sum(
            (2**grade - 1) / math.log2(rank + 1)
            for rank, grade in enumerate(values, start=1)
        )

    ideal = sorted(
        ideal_grades if ideal_grades is not None else grades,
        reverse=True,
    )
    ideal_dcg = dcg(ideal)
    return min(1.0, dcg(grades) / ideal_dcg) if ideal_dcg else 0.0


def evaluate_query(
    passages: Sequence[GoldPassage],
    chunks: Sequence[RetrievedChunk],
    *,
    relevance_threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate one ranked result list at the fixed cutoffs 5, 10, and 20."""

    if not passages:
        raise ValueError("at least one gold passage is required")
    if not 0 < relevance_threshold <= 1:
        raise ValueError("relevance_threshold must be in (0, 1]")

    metrics: dict[str, float] = {}
    hit_grades = [
        max(passage_coverage(chunk, passage) for passage in passages)
        for chunk in chunks
    ]
    relevant_hits = [grade >= relevance_threshold for grade in hit_grades]
    first_relevant_rank = next(
        (rank for rank, relevant in enumerate(relevant_hits, start=1) if relevant),
        None,
    )
    metrics["mrr"] = (
        1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
    )

    for cutoff in (5, 10, 20):
        top_chunks = chunks[:cutoff]
        found_passages = [
            any(
                passage_coverage(chunk, passage) >= relevance_threshold
                for chunk in top_chunks
            )
            for passage in passages
        ]
        complete_passages = [
            any(
                chunk.source_id == passage.source_id
                and chunk.char_start <= passage.char_start
                and chunk.char_end >= passage.char_end
                for chunk in top_chunks
            )
            for passage in passages
        ]
        metrics[f"recall_at_{cutoff}"] = _mean(
            float(found) for found in found_passages
        )
        metrics[f"passage_coverage_at_{cutoff}"] = _mean(
            _union_coverage(top_chunks, passage) for passage in passages
        )
        metrics[f"boundary_completeness_at_{cutoff}"] = _mean(
            float(complete) for complete in complete_passages
        )

    relevant_at_5 = sum(relevant_hits[:5])
    metrics["precision_at_5"] = relevant_at_5 / 5
    metrics["irrelevant_at_5"] = float(5 - relevant_at_5)
    metrics["ndcg_at_10"] = _ndcg(
        hit_grades[:10],
        [1.0] * min(len(passages), 10),
    )
    return metrics


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def aggregate_query_metrics(
    query_metrics: Sequence[Mapping[str, float]],
    latencies_ms: Sequence[float],
) -> dict[str, float | int]:
    """Macro-average query metrics and summarize request latency."""

    if not query_metrics:
        raise ValueError("at least one query result is required")
    metric_names = query_metrics[0].keys()
    summary: dict[str, float | int] = {
        "query_count": len(query_metrics),
        "request_count": len(latencies_ms),
    }
    for metric_name in metric_names:
        summary[metric_name] = _mean(
            float(metrics[metric_name]) for metrics in query_metrics
        )
    summary["latency_mean_ms"] = _mean(latencies_ms)
    summary["latency_p50_ms"] = _percentile(latencies_ms, 0.50)
    summary["latency_p95_ms"] = _percentile(latencies_ms, 0.95)
    return summary

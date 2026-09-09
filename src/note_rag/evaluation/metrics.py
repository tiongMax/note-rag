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

"""Deterministic retrieval, answer, citation, and refusal metrics."""

import math
import re
from collections import Counter

from note_rag.evaluation.models import (
    BenchmarkCase,
    EvaluationTrace,
    RetrievedContext,
)

_TOKEN = re.compile(r"\w+", re.UNICODE)
_CITATION = re.compile(r"\[(\d+)]")
_REFUSAL = re.compile(
    r"\b(?:cannot|can't|unable to|do not have|don't have|"
    r"not enough|insufficient|no (?:relevant )?(?:context|evidence|"
    r"information)|isn't (?:provided|available)|not (?:provided|available))\b",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    return " ".join(_TOKEN.findall(text.casefold()))


def token_f1(reference: str, answer: str) -> float:
    reference_tokens = _TOKEN.findall(reference.casefold())
    answer_tokens = _TOKEN.findall(answer.casefold())
    if not reference_tokens or not answer_tokens:
        return float(reference_tokens == answer_tokens)
    overlap = sum((Counter(reference_tokens) & Counter(answer_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(answer_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _matches(evidence_quote: str, chunk_text: str) -> bool:
    quote = normalize(evidence_quote)
    chunk = normalize(chunk_text)
    if not quote or not chunk:
        return False
    return quote in chunk or (chunk in quote and len(chunk) / len(quote) >= 0.8)


def chunk_relevance(
    case: BenchmarkCase,
    chunk: RetrievedContext,
) -> tuple[int, set[str]]:
    """Return maximum relevance grade and matched evidence groups."""

    grade = 0
    groups: set[str] = set()
    for evidence in case.evidence:
        if evidence.document and evidence.document != chunk.filename:
            continue
        id_match = chunk.chunk_id in evidence.chunk_ids
        if id_match or _matches(evidence.quote, chunk.text):
            grade = max(grade, evidence.relevance)
            groups.add(evidence.group)
    return grade, groups


def _ndcg(grades: list[int], ideal_grades: list[int]) -> float:
    def dcg(values: list[int]) -> float:
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

    ideal = dcg(ideal_grades)
    return dcg(grades) / ideal if ideal else 0.0


def retrieval_metrics(
    case: BenchmarkCase,
    chunks: list[RetrievedContext],
    *,
    cutoffs: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, float]:
    if not case.answerable:
        return {}
    groups = {evidence.group for evidence in case.evidence}
    group_grades = {
        group: max(
            evidence.relevance
            for evidence in case.evidence
            if evidence.group == group
        )
        for group in groups
    }
    labelled = [chunk_relevance(case, chunk) for chunk in chunks]
    metrics: dict[str, float] = {}
    first_relevant = next(
        (rank for rank, (grade, _) in enumerate(labelled, start=1) if grade),
        None,
    )
    metrics["mrr"] = 1.0 / first_relevant if first_relevant else 0.0
    for cutoff in cutoffs:
        selected = labelled[:cutoff]
        matched = set().union(*(item[1] for item in selected)) if selected else set()
        relevant_count = sum(1 for grade, _ in selected if grade)
        metrics[f"hit_rate@{cutoff}"] = float(relevant_count > 0)
        metrics[f"precision@{cutoff}"] = relevant_count / cutoff
        metrics[f"recall@{cutoff}"] = len(matched) / len(groups)
        metrics[f"complete_evidence@{cutoff}"] = float(matched == groups)
        grades = [grade for grade, _ in selected]
        ideal = sorted(group_grades.values(), reverse=True)[:cutoff]
        metrics[f"ndcg@{cutoff}"] = _ndcg(grades, ideal)
    return metrics


def evaluate_trace(
    case: BenchmarkCase,
    trace: EvaluationTrace,
) -> dict[str, float | int | None]:
    """Calculate all metrics that do not require an external judge."""

    metrics: dict[str, float | int | None] = dict(
        retrieval_metrics(case, trace.retrieved)
    )
    reference = case.reference_answer or ""
    metrics["exact_match"] = (
        float(normalize(reference) == normalize(trace.answer))
        if case.answerable
        else None
    )
    metrics["token_f1"] = (
        token_f1(reference, trace.answer) if case.answerable else None
    )
    cited_ids = [int(item) for item in _CITATION.findall(trace.answer)]
    available_ids = {chunk.citation_id for chunk in trace.retrieved}
    valid_count = sum(citation_id in available_ids for citation_id in cited_ids)
    metrics["citation_count"] = len(cited_ids)
    metrics["citation_validity"] = (
        valid_count / len(cited_ids) if cited_ids else float(not case.answerable)
    )
    metrics["has_citation"] = float(bool(cited_ids))
    refused = bool(_REFUSAL.search(trace.answer))
    expected_refusal = case.expected_behavior == "refuse"
    metrics["refusal_correct"] = float(refused == expected_refusal)
    metrics["false_refusal"] = float(refused and not expected_refusal)
    metrics["retrieval_ms"] = trace.retrieval_ms
    metrics["generation_ms"] = trace.generation_ms
    metrics["total_ms"] = trace.retrieval_ms + trace.generation_ms
    metrics["context_tokens"] = trace.context_tokens
    metrics["input_tokens"] = trace.input_tokens
    metrics["output_tokens"] = trace.output_tokens
    metrics["estimated_cost_usd"] = trace.estimated_cost_usd
    return metrics

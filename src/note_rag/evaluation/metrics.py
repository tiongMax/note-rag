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

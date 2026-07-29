"""Pure information-retrieval metric calculations."""

import math
from collections.abc import Sequence

from note_rag.evaluation.models import DEFAULT_EVAL_K, RankingMetrics


def ranking_metrics_at_k(
    returned_ids: Sequence[str],
    relevant_ids: Sequence[str],
    *,
    k: int = DEFAULT_EVAL_K,
) -> RankingMetrics:
    """Compute standard binary-relevance ranking metrics at k."""

    if k < 1:
        raise ValueError("k must be at least 1")
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("ranking metrics are undefined without relevant IDs")

    seen: set[str] = set()
    hit_flags: list[int] = []
    for chunk_id in returned_ids[:k]:
        is_new_relevant = chunk_id in relevant and chunk_id not in seen
        hit_flags.append(int(is_new_relevant))
        seen.add(chunk_id)

    hit_count = sum(hit_flags)
    precision = hit_count / k
    recall = hit_count / len(relevant)
    relevant_ranks = [
        rank for rank, is_relevant in enumerate(hit_flags, start=1) if is_relevant
    ]
    precision_sum = 0.0
    hits_so_far = 0
    for rank, is_relevant in enumerate(hit_flags, start=1):
        if is_relevant:
            hits_so_far += 1
            precision_sum += hits_so_far / rank

    dcg = sum(
        is_relevant / math.log2(rank + 1)
        for rank, is_relevant in enumerate(hit_flags, start=1)
    )
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_at_k": (
            2 * precision * recall / (precision + recall) if hit_count else 0.0
        ),
        "hit_rate_at_k": float(hit_count > 0),
        "reciprocal_rank": 1 / relevant_ranks[0] if relevant_ranks else 0.0,
        "average_precision_at_k": precision_sum / min(len(relevant), k),
        "ndcg_at_k": dcg / ideal_dcg,
    }


def precision_recall_at_5(
    returned_ids: list[str], relevant_ids: list[str]
) -> tuple[float, float]:
    """Backward-compatible precision@5 and recall@5 helper."""

    if not relevant_ids:
        return 0.0, 0.0
    metrics = ranking_metrics_at_k(returned_ids, relevant_ids)
    return metrics["precision_at_k"], metrics["recall_at_k"]


def relevant_ranks(
    returned_ids: Sequence[str], relevant_ids: Sequence[str], k: int
) -> list[int]:
    """Return one-based ranks for unique relevant results."""

    relevant = set(relevant_ids)
    seen: set[str] = set()
    ranks: list[int] = []
    for rank, chunk_id in enumerate(returned_ids[:k], start=1):
        if chunk_id in relevant and chunk_id not in seen:
            ranks.append(rank)
            seen.add(chunk_id)
    return ranks

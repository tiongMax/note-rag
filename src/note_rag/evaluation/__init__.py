"""Retrieval evaluation metrics grounded in source-document spans."""

from note_rag.evaluation.metrics import (
    GoldPassage,
    RetrievedChunk,
    aggregate_query_metrics,
    evaluate_query,
    passage_coverage,
)
from note_rag.evaluation.ranking import Bm25Index, weighted_rrf

__all__ = [
    "GoldPassage",
    "RetrievedChunk",
    "aggregate_query_metrics",
    "evaluate_query",
    "passage_coverage",
    "Bm25Index",
    "weighted_rrf",
]

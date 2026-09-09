"""Retrieval evaluation metrics grounded in source-document spans and reproducible framework-independent RAG evaluation."""

from note_rag.evaluation.dataset import load_benchmark
from note_rag.evaluation.metrics import (
    GoldPassage,
    RetrievedChunk,
    aggregate_query_metrics,
    evaluate_query,
    passage_coverage,
    evaluate_trace
)
from note_rag.evaluation.ranking import Bm25Index, weighted_rrf
from note_rag.evaluation.models import (
    BenchmarkCase,
    EvaluationTrace,
    Evidence,
    RetrievedContext,
)

__all__ = [
    "GoldPassage",
    "RetrievedChunk",
    "aggregate_query_metrics",
    "evaluate_query",
    "passage_coverage",
    "Bm25Index",
    "weighted_rrf",
    "BenchmarkCase",
    "EvaluationTrace",
    "Evidence",
    "RetrievedContext",
    "evaluate_trace",
    "load_benchmark",
]

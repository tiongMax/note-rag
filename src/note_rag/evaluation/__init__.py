"""Retrieval evaluation metrics and reproducible RAG evaluation."""

from note_rag.evaluation.dataset import load_benchmark
from note_rag.evaluation.education import (
    citation_support,
    difficulty_consistency,
    duplicate_question_rate,
    evaluate_learning_materials,
    multiple_choice_quality,
)
from note_rag.evaluation.metrics import (
    GoldPassage,
    RetrievedChunk,
    aggregate_query_metrics,
    evaluate_query,
    evaluate_trace,
    passage_coverage,
)
from note_rag.evaluation.models import (
    BenchmarkCase,
    EvaluationTrace,
    Evidence,
    RetrievedContext,
)
from note_rag.evaluation.ranking import Bm25Index, weighted_rrf

__all__ = [
    "GoldPassage",
    "RetrievedChunk",
    "aggregate_query_metrics",
    "evaluate_query",
    "passage_coverage",
    "citation_support",
    "difficulty_consistency",
    "duplicate_question_rate",
    "evaluate_learning_materials",
    "multiple_choice_quality",
    "Bm25Index",
    "weighted_rrf",
    "BenchmarkCase",
    "EvaluationTrace",
    "Evidence",
    "RetrievedContext",
    "evaluate_trace",
    "load_benchmark",
]

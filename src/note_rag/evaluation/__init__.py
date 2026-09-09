"""Reproducible, framework-independent RAG evaluation."""

from note_rag.evaluation.dataset import load_benchmark
from note_rag.evaluation.metrics import evaluate_trace
from note_rag.evaluation.models import (
    BenchmarkCase,
    EvaluationTrace,
    Evidence,
    RetrievedContext,
)

__all__ = [
    "BenchmarkCase",
    "EvaluationTrace",
    "Evidence",
    "RetrievedContext",
    "evaluate_trace",
    "load_benchmark",
]

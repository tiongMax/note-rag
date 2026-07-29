"""Public evaluation API."""

from note_rag.evaluation.generation import evaluate_generation
from note_rag.evaluation.loader import load_eval_set
from note_rag.evaluation.metrics import precision_recall_at_5, ranking_metrics_at_k
from note_rag.evaluation.models import (
    DEFAULT_EVAL_K,
    EVAL_CATEGORIES,
    AnswerGenerator,
    EvalCase,
    GenerationResult,
    GenerationScores,
    GenerationSummary,
    QuestionResult,
    RagJudge,
    RankingMetrics,
    SummaryResult,
)
from note_rag.evaluation.reporting import (
    print_comparison_table,
    print_generation_table,
    save_comparison_csv,
    save_generation_csvs,
    save_retrieval_details_csv,
)
from note_rag.evaluation.retrieval import evaluate_retrievers

__all__ = [
    "DEFAULT_EVAL_K",
    "EVAL_CATEGORIES",
    "AnswerGenerator",
    "EvalCase",
    "GenerationResult",
    "GenerationScores",
    "GenerationSummary",
    "QuestionResult",
    "RagJudge",
    "RankingMetrics",
    "SummaryResult",
    "evaluate_generation",
    "evaluate_retrievers",
    "load_eval_set",
    "precision_recall_at_5",
    "print_comparison_table",
    "print_generation_table",
    "ranking_metrics_at_k",
    "save_comparison_csv",
    "save_generation_csvs",
    "save_retrieval_details_csv",
]

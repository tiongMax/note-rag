"""Data contracts for RAG evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NotRequired, Protocol, TypedDict

from langchain_core.documents import Document

EVAL_CATEGORIES = {"single-chunk", "multi-chunk", "out-of-corpus"}
DEFAULT_EVAL_K = 5


class EvalCase(TypedDict):
    id: NotRequired[str]
    question: str
    answer: str
    relevant_chunk_ids: list[str]
    category: str


class RankingMetrics(TypedDict):
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    hit_rate_at_k: float
    reciprocal_rank: float
    average_precision_at_k: float
    ndcg_at_k: float


class QuestionResult(TypedDict):
    case_id: str
    config: str
    question: str
    category: str
    relevant_chunk_ids: list[str]
    returned_chunk_ids: list[str]
    relevant_ranks: list[int]
    precision_at_k: float | None
    recall_at_k: float | None
    f1_at_k: float | None
    hit_rate_at_k: float | None
    reciprocal_rank: float | None
    average_precision_at_k: float | None
    ndcg_at_k: float | None


class SummaryResult(TypedDict):
    config: str
    slice: str
    query_count: int
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    hit_rate_at_k: float
    mrr: float
    map_at_k: float
    ndcg_at_k: float


class GenerationScores(TypedDict):
    correctness: float
    faithfulness: float
    answer_relevance: float
    context_relevance: float
    refusal_correctness: float | None
    explanation: str


class GenerationResult(TypedDict):
    case_id: str
    config: str
    question: str
    category: str
    reference_answer: str
    generated_answer: str
    returned_chunk_ids: list[str]
    correctness: float
    faithfulness: float
    answer_relevance: float
    context_relevance: float
    refusal_correctness: float | None
    explanation: str


class GenerationSummary(TypedDict):
    config: str
    slice: str
    query_count: int
    correctness: float
    faithfulness: float
    answer_relevance: float
    context_relevance: float
    refusal_correctness: float | None


class RagJudge(Protocol):
    def __call__(
        self,
        *,
        question: str,
        reference_answer: str,
        generated_answer: str,
        contexts: Sequence[str],
        category: str,
    ) -> GenerationScores: ...


AnswerGenerator = Callable[[str, Sequence[Document]], str]

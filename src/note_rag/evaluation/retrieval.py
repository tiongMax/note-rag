"""Ranked retrieval evaluation."""

from collections.abc import Sequence
from statistics import fmean
from typing import cast

from note_rag.evaluation.metrics import ranking_metrics_at_k, relevant_ranks
from note_rag.evaluation.models import (
    DEFAULT_EVAL_K,
    EvalCase,
    QuestionResult,
    SummaryResult,
)
from note_rag.retrieval import RetrievalPipeline, document_chunk_id


def _mean_metrics(
    config_name: str, slice_name: str, results: Sequence[QuestionResult]
) -> SummaryResult:
    metric_names = (
        "precision_at_k",
        "recall_at_k",
        "f1_at_k",
        "hit_rate_at_k",
        "reciprocal_rank",
        "average_precision_at_k",
        "ndcg_at_k",
    )
    means = {
        name: fmean(cast("float", result[name]) for result in results)
        for name in metric_names
    }
    return {
        "config": config_name,
        "slice": slice_name,
        "query_count": len(results),
        "precision_at_k": means["precision_at_k"],
        "recall_at_k": means["recall_at_k"],
        "f1_at_k": means["f1_at_k"],
        "hit_rate_at_k": means["hit_rate_at_k"],
        "mrr": means["reciprocal_rank"],
        "map_at_k": means["average_precision_at_k"],
        "ndcg_at_k": means["ndcg_at_k"],
    }


def _evaluate_case(
    case: EvalCase,
    index: int,
    config_name: str,
    pipeline: RetrievalPipeline,
    k: int,
) -> QuestionResult:
    documents = pipeline.retrieve(case["question"])
    returned_ids = [document_chunk_id(document) for document in documents]
    relevant_ids = case["relevant_chunk_ids"]
    metrics = (
        ranking_metrics_at_k(returned_ids, relevant_ids, k=k) if relevant_ids else None
    )

    def value(name: str) -> float | None:
        return metrics[name] if metrics is not None else None  # type: ignore[literal-required]

    return {
        "case_id": case.get("id", f"q{index:03d}"),
        "config": config_name,
        "question": case["question"],
        "category": case["category"],
        "relevant_chunk_ids": relevant_ids,
        "returned_chunk_ids": returned_ids[:k],
        "relevant_ranks": relevant_ranks(returned_ids, relevant_ids, k),
        "precision_at_k": value("precision_at_k"),
        "recall_at_k": value("recall_at_k"),
        "f1_at_k": value("f1_at_k"),
        "hit_rate_at_k": value("hit_rate_at_k"),
        "reciprocal_rank": value("reciprocal_rank"),
        "average_precision_at_k": value("average_precision_at_k"),
        "ndcg_at_k": value("ndcg_at_k"),
    }


def evaluate_retrievers(
    cases: list[EvalCase],
    pipelines: dict[str, RetrievalPipeline],
    *,
    k: int = DEFAULT_EVAL_K,
) -> tuple[list[QuestionResult], list[SummaryResult]]:
    """Evaluate ranked retrieval across configurations and dataset slices."""

    all_results: list[QuestionResult] = []
    summaries: list[SummaryResult] = []
    for config_name, pipeline in pipelines.items():
        results = [
            _evaluate_case(case, index, config_name, pipeline, k)
            for index, case in enumerate(cases, start=1)
        ]
        all_results.extend(results)
        slices = {
            "in-corpus": [
                result for result in results if result["category"] != "out-of-corpus"
            ],
            "single-chunk": [
                result for result in results if result["category"] == "single-chunk"
            ],
            "multi-chunk": [
                result for result in results if result["category"] == "multi-chunk"
            ],
        }
        summaries.extend(
            _mean_metrics(config_name, name, values)
            for name, values in slices.items()
            if values
        )
    return all_results, summaries

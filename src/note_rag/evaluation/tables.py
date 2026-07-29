"""Console tables for evaluation results."""

from collections.abc import Sequence

from note_rag.evaluation.models import GenerationSummary, SummaryResult


def print_comparison_table(summaries: list[SummaryResult]) -> None:
    """Print the rank-aware retrieval comparison."""

    headers = (
        "config",
        "slice",
        "n",
        "P@5",
        "R@5",
        "F1@5",
        "Hit@5",
        "MRR",
        "MAP@5",
        "nDCG@5",
    )
    rows = [
        (
            row["config"],
            row["slice"],
            str(row["query_count"]),
            f"{row['precision_at_k']:.4f}",
            f"{row['recall_at_k']:.4f}",
            f"{row['f1_at_k']:.4f}",
            f"{row['hit_rate_at_k']:.4f}",
            f"{row['mrr']:.4f}",
            f"{row['map_at_k']:.4f}",
            f"{row['ndcg_at_k']:.4f}",
        )
        for row in summaries
    ]
    _print_table(headers, rows)


def print_generation_table(summaries: list[GenerationSummary]) -> None:
    """Print the end-to-end answer-quality comparison."""

    headers = (
        "config",
        "slice",
        "n",
        "correct",
        "faithful",
        "answer-rel",
        "context-rel",
        "refusal",
    )
    rows = [
        (
            row["config"],
            row["slice"],
            str(row["query_count"]),
            f"{row['correctness']:.4f}",
            f"{row['faithfulness']:.4f}",
            f"{row['answer_relevance']:.4f}",
            f"{row['context_relevance']:.4f}",
            (
                f"{row['refusal_correctness']:.4f}"
                if row["refusal_correctness"] is not None
                else "-"
            ),
        )
        for row in summaries
    ]
    _print_table(headers, rows)


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))

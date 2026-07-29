"""Command-line evaluation workflow."""

from pathlib import Path
from typing import Protocol

from langchain_postgres import PGVector

from note_rag.config import Settings
from note_rag.evaluation import (
    evaluate_generation,
    evaluate_retrievers,
    load_eval_set,
    print_comparison_table,
    print_generation_table,
    save_comparison_csv,
    save_generation_csvs,
    save_retrieval_details_csv,
)
from note_rag.rag import build_evaluation_answer_generator, build_rag_judge
from note_rag.retrieval import RetrievalStrategy
from note_rag.retrieval.factory import build_retrieval_pipelines


class EvaluationOptions(Protocol):
    eval_set: Path | None
    eval_output: Path
    eval_generation: bool
    strategy: RetrievalStrategy


def run_evaluation(
    vector_store: PGVector, args: EvaluationOptions, settings: Settings
) -> None:
    """Run retrieval and optional generation evaluations."""

    assert args.eval_set is not None
    cases = load_eval_set(args.eval_set)
    pipelines = build_retrieval_pipelines(vector_store, settings)
    details, summaries = evaluate_retrievers(cases, pipelines)
    print_comparison_table(summaries)
    save_comparison_csv(args.eval_output, summaries)
    detail_path = save_retrieval_details_csv(args.eval_output, details)
    print(f"\nSaved retrieval summary to {args.eval_output.resolve()}")
    print(f"Saved per-question details to {detail_path.resolve()}")
    if not args.eval_generation:
        return
    print(
        f"\nEnd-to-end generation evaluation: {args.strategy.value} "
        f"(judge: {settings.eval_judge_model})"
    )
    results, generation_summaries = evaluate_generation(
        cases,
        config_name=args.strategy.value,
        pipeline=pipelines[args.strategy.value],
        answer_generator=build_evaluation_answer_generator(settings),
        judge=build_rag_judge(settings),
    )
    print_generation_table(generation_summaries)
    detail_path, summary_path = save_generation_csvs(
        args.eval_output, results, generation_summaries
    )
    print(f"Saved generation details to {detail_path.resolve()}")
    print(f"Saved generation summary to {summary_path.resolve()}")

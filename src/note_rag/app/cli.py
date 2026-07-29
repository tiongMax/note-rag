"""Command-line interface for the Note RAG application."""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_core.runnables import Runnable

from note_rag.app.evaluation import run_evaluation
from note_rag.config import PROJECT_ROOT, settings
from note_rag.ingestion import ingest_pdfs
from note_rag.rag import RagResult, build_rag_chain
from note_rag.retrieval import RetrievalStrategy
from note_rag.retrieval.factory import (
    build_retrieval_pipeline,
    create_embeddings,
    create_vector_store,
)


class CliArgs(argparse.Namespace):
    """Typed command-line arguments."""

    pdfs: list[Path]
    strategy: RetrievalStrategy
    eval_set: Path | None
    eval_output: Path
    eval_generation: bool


def pdf_path(value: str) -> Path:
    """Validate a positional PDF argument."""

    path = Path(value)
    if path.suffix.lower() != ".pdf":
        raise argparse.ArgumentTypeError("input must be a .pdf file")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"PDF not found: {path}")
    return path.resolve()


def parse_args() -> CliArgs:
    """Parse ingestion, retrieval, and evaluation options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pdfs",
        nargs="*",
        type=pdf_path,
        metavar="PDF",
        help="Optional local PDF files to ingest sequentially before chat",
    )
    parser.add_argument(
        "--strategy",
        type=RetrievalStrategy,
        choices=list(RetrievalStrategy),
        default=RetrievalStrategy.HYBRID_RERANK,
        help="Retrieval strategy for interactive chat (default: hybrid+rerank)",
    )
    parser.add_argument(
        "--eval-set",
        type=Path,
        help="Evaluate all three strategies using this JSON file, then exit",
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=PROJECT_ROOT / "evaluations" / "results" / "retrieval_eval.csv",
        help=(
            "Aggregate evaluation CSV path "
            "(default: evaluations/results/retrieval_eval.csv)"
        ),
    )
    parser.add_argument(
        "--eval-generation",
        action="store_true",
        help="Also generate and LLM-judge answers (incurs Gemini calls)",
    )
    return parser.parse_args(namespace=CliArgs())


def chat(chain: Runnable[str, RagResult]) -> None:
    """Run the interactive question loop."""

    print("\nPDF RAG is ready. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return
        if not question:
            continue
        result = chain.invoke(question)
        print(f"\nAnswer:\n{result['answer']}")


def _ingest_requested_pdfs(vector_store: object, args: CliArgs) -> None:
    if not args.pdfs:
        print("No PDFs supplied; using the existing PGVector collection.")
        return
    results = ingest_pdfs(vector_store, args.pdfs)  # type: ignore[arg-type]
    for pdf, chunk_count in results:
        print(f"Upserted {chunk_count} chunks from {pdf.name} into PGVector.")
    if len(results) > 1:
        print(
            f"Batch ingestion complete: {len(results)} PDFs, "
            f"{sum(count for _, count in results)} chunks."
        )


def main() -> int:
    """Ingest PDFs, evaluate retrieval, or start interactive chat."""

    args = parse_args()
    try:
        vector_store = create_vector_store(create_embeddings(settings), settings)
        _ingest_requested_pdfs(vector_store, args)
        if args.eval_set is not None:
            run_evaluation(vector_store, args, settings)
            return 0
        pipeline = build_retrieval_pipeline(vector_store, args.strategy, settings)
        print(f"Retrieval strategy: {args.strategy.value}")
        chat(build_rag_chain(pipeline.as_runnable(), settings))
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

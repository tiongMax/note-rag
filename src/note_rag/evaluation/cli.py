"""Command-line orchestration for reproducible RAG experiments."""

import argparse
import hashlib
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from note_rag.config import load_environment
from note_rag.evaluation.client import NoteRagHttpClient, RetrievalOptions
from note_rag.evaluation.dataset import load_benchmark, load_traces
from note_rag.evaluation.metrics import evaluate_trace
from note_rag.evaluation.models import (
    EvaluationTrace,
    ExperimentSnapshot,
    QuestionResult,
)
from note_rag.evaluation.reporting import write_reports


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="note-rag-eval",
        description="Evaluate a running Note RAG experiment.",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--base-url",
        default=os.getenv("NOTE_RAG_EVAL_BASE_URL", "http://127.0.0.1:8001"),
    )
    parser.add_argument("--api-token", default=os.getenv("API_AUTH_TOKEN", ""))
    parser.add_argument("--trace-input", type=Path)
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        help="File whose SHA-256 identifies the frozen evaluation corpus.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--mode",
        choices=("vector", "keyword", "hybrid"),
        default="hybrid",
    )
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--max-chunks", type=int, default=8)
    parser.add_argument("--max-context-tokens", type=int, default=1200)
    parser.add_argument("--vector-weight", type=float, default=0.7)
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--rerank-weight", type=float, default=0.7)
    parser.add_argument("--input-cost-per-million", type=float)
    parser.add_argument("--output-cost-per-million", type=float)
    parser.add_argument("--ragas", action="store_true")
    parser.add_argument(
        "--judge-model",
        default=os.getenv("RAGAS_JUDGE_MODEL", "gemini-3.5-flash"),
    )
    parser.add_argument(
        "--judge-embedding-model",
        default=os.getenv(
            "RAGAS_EMBEDDING_MODEL",
            "gemini-embedding-001",
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive = {
        "--candidate-k": args.candidate_k,
        "--max-chunks": args.max_chunks,
        "--max-context-tokens": args.max_context_tokens,
        "--timeout-seconds": args.timeout_seconds,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"{name} must be greater than zero")
    for name, value in (
        ("--vector-weight", args.vector_weight),
        ("--rerank-weight", args.rerank_weight),
    ):
        if not 0 <= value <= 1:
            parser.error(f"{name} must be between zero and one")
    if not args.experiment.strip():
        parser.error("--experiment cannot be empty")
    if any(character in args.experiment for character in '<>:"/\\|?*'):
        parser.error("--experiment must be a filesystem-safe name")
    cost_values = (
        args.input_cost_per_million,
        args.output_cost_per_million,
    )
    if sum(value is not None for value in cost_values) == 1:
        parser.error("both input and output token costs must be provided")
    if any(value is not None and value < 0 for value in cost_values):
        parser.error("token costs cannot be negative")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _error_trace(case_id: str, error: Exception) -> EvaluationTrace:
    return EvaluationTrace(
        question_id=case_id,
        answer="",
        retrieval_ms=0,
        generation_ms=0,
        error=f"{type(error).__name__}: {error}",
    )


def _print_summary(report: dict[str, Any], output_dir: Path) -> None:
    print(f"Evaluated {report['questions']} questions")
    print(f"Execution errors: {report['execution_errors']}")
    for name, summary in report["metrics"].items():
        print(f"{name}: {summary['mean']:.4f} (n={summary['count']})")
    print(f"Reports: {output_dir.resolve()}")


def main(argv: list[str] | None = None) -> int:
    load_environment()
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    cases = load_benchmark(args.dataset)
    options = RetrievalOptions(
        mode=args.mode,
        candidate_k=args.candidate_k,
        max_chunks=args.max_chunks,
        max_context_tokens=args.max_context_tokens,
        vector_weight=args.vector_weight,
        rerank=args.rerank,
        rerank_weight=args.rerank_weight,
    )
    saved_traces = load_traces(args.trace_input) if args.trace_input else None
    client = NoteRagHttpClient(
        base_url=args.base_url,
        api_token=args.api_token,
        timeout_seconds=args.timeout_seconds,
        options=options,
        input_cost_per_million=args.input_cost_per_million,
        output_cost_per_million=args.output_cost_per_million,
    )
    output_dir = args.output_dir / args.experiment
    judge = None
    if args.ragas:
        from note_rag.evaluation.ragas_adapter import RagasEvaluator

        judge = RagasEvaluator(
            api_key=(
                os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_API_KEY")
                or ""
            ),
            judge_model=args.judge_model,
            embedding_model=args.judge_embedding_model,
            cache_dir=output_dir / ".ragas-cache",
        )

    results: list[QuestionResult] = []
    for position, case in enumerate(cases, start=1):
        print(f"[{position}/{len(cases)}] {case.id}", file=sys.stderr)
        try:
            if saved_traces is not None:
                trace = saved_traces[case.id]
            else:
                trace = client.run(case)
        except (KeyError, RuntimeError, ValueError) as error:
            if args.fail_fast:
                raise
            trace = _error_trace(case.id, error)
        judge_scores = judge.evaluate(case, trace) if judge else {}
        results.append(
            QuestionResult(
                case=case,
                trace=trace,
                metrics=evaluate_trace(case, trace),
                judge=judge_scores,
            )
        )

    snapshot = ExperimentSnapshot(
        experiment=args.experiment,
        benchmark_path=str(args.dataset.resolve()),
        benchmark_sha256=_sha256(args.dataset),
        benchmark_cases=len(cases),
        corpus_manifest=(
            str(args.corpus_manifest.resolve())
            if args.corpus_manifest
            else None
        ),
        corpus_sha256=(
            _sha256(args.corpus_manifest) if args.corpus_manifest else None
        ),
        base_url=None if saved_traces is not None else args.base_url,
        git_commit=_git_commit(),
        created_at=datetime.now(UTC).isoformat(),
        retrieval={
            **asdict(options),
            "trace_input": (
                str(args.trace_input.resolve()) if args.trace_input else None
            ),
            "input_cost_per_million": args.input_cost_per_million,
            "output_cost_per_million": args.output_cost_per_million,
        },
        ragas={
            "enabled": bool(args.ragas),
            "version": version("ragas") if args.ragas else None,
            "judge_model": args.judge_model if args.ragas else None,
            "embedding_model": (
                args.judge_embedding_model if args.ragas else None
            ),
        },
    )
    report = write_reports(output_dir, snapshot, results)
    _print_summary(report, output_dir)
    return int(report["execution_errors"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())

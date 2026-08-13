"""Collect grounded answers and evaluate them with RAGAS."""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from run_retrieval_benchmark import ApiClient  # noqa: E402

from note_rag.evaluation import (  # noqa: E402
    aggregate_ragas_scores,
    build_run_metadata,
    load_jsonl,
    summarize_latencies,
    validate_run_label,
    write_evaluation_artifacts,
)

_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "context_precision",
    "context_recall",
)


class RagasScorer(Protocol):
    def score(self, **kwargs: Any) -> Any: ...


def collect_samples(
    *,
    client: ApiClient,
    entries: Sequence[Mapping[str, Any]],
    mode: str,
    candidate_k: int,
    max_chunks: int,
    max_context_tokens: int,
    vector_weight: float,
    rerank: bool,
    rerank_weight: float | None,
    delay_ms: float = 0,
) -> list[dict[str, Any]]:
    """Generate answers and capture contexts under one fixed configuration."""

    filenames = sorted(
        {
            str(passage["source_id"])
            for entry in entries
            for passage in entry["relevant_passages"]
        }
    )
    options: dict[str, Any] = {
        "mode": mode,
        "candidate_k": candidate_k,
        "max_chunks": max_chunks,
        "max_context_tokens": max_context_tokens,
        "vector_weight": vector_weight,
        "rerank": rerank,
        "filters": {"filenames": filenames},
    }
    if rerank_weight is not None:
        options["rerank_weight"] = rerank_weight

    samples = []
    for number, entry in enumerate(entries, start=1):
        payload = {"query": entry["query"], **options}
        total_started = time.perf_counter()
        context_started = time.perf_counter()
        context = client.request("POST", "/api/v1/retrieval/context", payload)
        context_latency_ms = (time.perf_counter() - context_started) * 1000
        chat_started = time.perf_counter()
        chat = client.request("POST", "/api/v1/chat", payload)
        chat_latency_ms = (time.perf_counter() - chat_started) * 1000
        total_latency_ms = (time.perf_counter() - total_started) * 1000

        chunks = context.get("chunks")
        if not isinstance(chunks, list):
            raise RuntimeError("context response has no chunk list")
        answer = chat.get("answer")
        if not isinstance(answer, str):
            raise RuntimeError("chat response has no answer")
        samples.append(
            {
                "id": entry["id"],
                "user_input": entry["query"],
                "response": answer,
                "reference": entry["expected_answer"],
                "retrieved_contexts": [str(chunk["text"]) for chunk in chunks],
                "tags": entry.get("tags", []),
                "answer_model": chat.get("model_name"),
                "conversation_id": chat.get("conversation_id"),
                "citations": chat.get("citations", []),
                "context_chunk_ids": [str(chunk["chunk_id"]) for chunk in chunks],
                "context_token_count": context.get("token_count"),
                "context_truncated": context.get("truncated"),
                "context_latency_ms": context_latency_ms,
                "chat_latency_ms": chat_latency_ms,
                "total_latency_ms": total_latency_ms,
            }
        )
        print(
            f"[{number:02}/{len(entries)}] {entry['id']}: "
            f"answer + {len(chunks)} contexts ({total_latency_ms:.1f} ms)"
        )
        if delay_ms and number < len(entries):
            time.sleep(delay_ms / 1000)
    return samples


def validate_samples(
    samples: Sequence[Mapping[str, Any]],
    entries: Sequence[Mapping[str, Any]],
) -> None:
    """Ensure saved samples correspond exactly to the selected gold entries."""

    expected_ids = [str(entry["id"]) for entry in entries]
    actual_ids = [str(sample.get("id", "")) for sample in samples]
    if actual_ids != expected_ids:
        raise ValueError("sample IDs and order do not match the selected dataset")
    required = {"user_input", "response", "reference", "retrieved_contexts"}
    for sample in samples:
        missing = sorted(required - sample.keys())
        if missing:
            rendered = ", ".join(missing)
            raise ValueError(f"sample {sample.get('id')!r} is missing: {rendered}")


def summarize_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {"sample_count": len(samples)}
    for field, prefix in (
        ("context_latency_ms", "context_latency"),
        ("chat_latency_ms", "chat_latency"),
        ("total_latency_ms", "total_latency"),
    ):
        values = [
            float(sample[field])
            for sample in samples
            if sample.get(field) is not None
        ]
        metrics.update(summarize_latencies(values, prefix=prefix))
    token_counts = [
        float(sample["context_token_count"])
        for sample in samples
        if sample.get("context_token_count") is not None
    ]
    metrics["context_tokens_mean"] = (
        statistics.fmean(token_counts) if token_counts else 0.0
    )
    metrics["context_truncated_count"] = sum(
        sample.get("context_truncated") is True for sample in samples
    )
    return metrics


def create_scorers(
    *,
    metric_names: Sequence[str],
    api_key: str,
    evaluator_model: str,
    embedding_model: str,
) -> dict[str, RagasScorer]:
    """Create modern RAGAS collection metrics backed by Google Gemini."""

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required")
    try:
        from google import genai
        from ragas.embeddings import GoogleEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerCorrectness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as error:
        raise RuntimeError(
            "RAGAS dependencies are missing; install with "
            'pip install -e ".[evaluation]"'
        ) from error

    client = genai.Client(api_key=api_key)
    llm = llm_factory(evaluator_model, provider="google", client=client)
    embeddings = GoogleEmbeddings(client=client, model=embedding_model)
    factories = {
        "faithfulness": lambda: Faithfulness(llm=llm),
        "answer_relevancy": lambda: AnswerRelevancy(llm=llm, embeddings=embeddings),
        "answer_correctness": lambda: AnswerCorrectness(llm=llm, embeddings=embeddings),
        "context_precision": lambda: ContextPrecision(llm=llm),
        "context_recall": lambda: ContextRecall(llm=llm),
    }
    return {name: factories[name]() for name in metric_names}


def evaluate_samples(
    samples: Sequence[Mapping[str, Any]],
    scorers: Mapping[str, RagasScorer],
) -> list[dict[str, Any]]:
    """Score samples while retaining per-metric judge reasoning."""

    metric_inputs = {
        "faithfulness": ("user_input", "response", "retrieved_contexts"),
        "answer_relevancy": ("user_input", "response"),
        "answer_correctness": ("response", "reference"),
        "context_precision": ("user_input", "reference", "retrieved_contexts"),
        "context_recall": ("user_input", "reference", "retrieved_contexts"),
    }
    results = []
    for number, sample in enumerate(samples, start=1):
        scores: dict[str, float] = {}
        reasons: dict[str, Any] = {}
        for name, scorer in scorers.items():
            kwargs = {key: sample[key] for key in metric_inputs[name]}
            result = scorer.score(**kwargs)
            scores[name] = float(result.value)
            reason = getattr(result, "reason", None)
            if reason is not None:
                reasons[name] = reason
        results.append({**sample, "ragas": scores, "ragas_reasons": reasons})
        rendered = ", ".join(f"{name}={value:.3f}" for name, value in scores.items())
        print(f"[{number:02}/{len(samples)}] {sample['id']}: {rendered}")
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect Note RAG answers and optionally judge them with RAGAS."
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPOSITORY_ROOT / "benchmarks/gold/dataset.jsonl",
    )
    parser.add_argument("--samples", type=Path, help="judge an existing sample JSONL")
    parser.add_argument(
        "--output-dir", type=Path, default=REPOSITORY_ROOT / "tmp/benchmarks"
    )
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--mode",
        choices=("vector", "keyword", "hybrid"),
        default="hybrid",
    )
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--max-chunks", type=int, default=8)
    parser.add_argument("--max-context-tokens", type=int, default=1800)
    parser.add_argument("--vector-weight", type=float, default=0.7)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--rerank-weight", type=float)
    parser.add_argument("--delay-ms", type=float, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--api-token-env", default="API_AUTH_TOKEN")
    parser.add_argument(
        "--evaluator-model",
        default=os.getenv("RAGAS_EVALUATOR_MODEL", "gemini-2.5-flash"),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("RAGAS_EMBEDDING_MODEL", "gemini-embedding-001"),
    )
    parser.add_argument(
        "--metrics", nargs="+", choices=_METRIC_NAMES, default=list(_METRIC_NAMES)
    )
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        validate_run_label(args.label)
    except ValueError as error:
        parser.error(str(error))
    if args.samples is not None and args.collect_only:
        parser.error("--samples and --collect-only cannot be used together")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")
    if not 1 <= args.candidate_k <= 100:
        parser.error("--candidate-k must be between 1 and 100")
    if not 1 <= args.max_chunks <= 100:
        parser.error("--max-chunks must be between 1 and 100")
    if args.max_context_tokens <= 0:
        parser.error("--max-context-tokens must be greater than zero")
    if not 0 <= args.vector_weight <= 1:
        parser.error("--vector-weight must be between 0 and 1")
    if args.rerank_weight is not None and not 0 <= args.rerank_weight <= 1:
        parser.error("--rerank-weight must be between 0 and 1")
    if args.delay_ms < 0:
        parser.error("--delay-ms cannot be negative")

    dataset_path = args.dataset.resolve()
    entries = load_jsonl(dataset_path)
    if args.limit is not None:
        entries = entries[: args.limit]
    configuration = {
        "base_url": args.base_url.rstrip("/"),
        "mode": args.mode,
        "candidate_k": args.candidate_k,
        "max_chunks": args.max_chunks,
        "max_context_tokens": args.max_context_tokens,
        "vector_weight": args.vector_weight,
        "rerank": not args.no_rerank,
        "rerank_weight": args.rerank_weight,
    }
    metadata = build_run_metadata(
        label=args.label, dataset_path=dataset_path, configuration=configuration
    )
    output_dir = args.output_dir.resolve()

    if args.samples is not None:
        samples = load_jsonl(args.samples.resolve())
        if args.limit is not None:
            samples = samples[: args.limit]
    else:
        token = os.getenv(args.api_token_env, "") if args.api_token_env else ""
        client = ApiClient(args.base_url, token=token, timeout=args.timeout_seconds)
        samples = collect_samples(
            client=client,
            entries=entries,
            mode=args.mode,
            candidate_k=args.candidate_k,
            max_chunks=args.max_chunks,
            max_context_tokens=args.max_context_tokens,
            vector_weight=args.vector_weight,
            rerank=not args.no_rerank,
            rerank_weight=args.rerank_weight,
            delay_ms=args.delay_ms,
        )
        sample_paths = write_evaluation_artifacts(
            output_dir=output_dir,
            label=args.label,
            kind="rag-samples",
            results=samples,
            metadata={**metadata, "sample_count": len(samples)},
            metrics=summarize_samples(samples),
        )
        print("Wrote samples:")
        for path in sample_paths:
            print(f"- {path}")

    validate_samples(samples, entries)
    if args.collect_only:
        return

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    scorers = create_scorers(
        metric_names=args.metrics,
        api_key=api_key,
        evaluator_model=args.evaluator_model,
        embedding_model=args.embedding_model,
    )
    results = evaluate_samples(samples, scorers)
    metrics = aggregate_ragas_scores([result["ragas"] for result in results])
    paths = write_evaluation_artifacts(
        output_dir=output_dir,
        label=args.label,
        kind="ragas",
        results=results,
        metadata={
            **metadata,
            "sample_count": len(results),
            "metrics": list(args.metrics),
            "evaluator_provider": "google",
            "evaluator_model": args.evaluator_model,
            "embedding_model": args.embedding_model,
        },
        metrics=metrics,
    )
    print(json.dumps(metrics, indent=2))
    print("Wrote RAGAS results:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()

"""Benchmark exact-query cache behavior against a running Note RAG API."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Measurement:
    request_index: int
    query_id: str
    normalized_query: str
    latency_ms: float
    status_code: int
    embedding_cache_status: str
    retrieval_cache_status: str
    corpus_version: str
    result_chunk_ids: str
    error: str


def main() -> None:
    from note_rag.config import load_environment

    load_environment()
    args = _parse_args()
    if args.clear_cache:
        _clear_cache()
    queries = _read_queries(args.queries)
    trace = _build_trace(
        queries,
        request_count=args.requests,
        repeat_rate=args.repeat_rate,
        seed=args.seed,
    )
    headers = {"Content-Type": "application/json"}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"

    metrics_before = _fetch_metrics(args.base_url, headers)
    started = time.perf_counter()
    measurements: list[Measurement] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                _execute,
                args.base_url,
                query,
                index,
                headers,
                args.timeout,
                args.mode,
                args.top_k,
                args.vector_weight,
                args.filters,
            ): index
            for index, query in enumerate(trace)
        }
        for future in as_completed(futures):
            measurements.append(future.result())
    elapsed = time.perf_counter() - started
    metrics_after = _fetch_metrics(args.base_url, headers)
    measurements.sort(key=lambda row: row.request_index)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.label}-repeat-{int(args.repeat_rate * 100):02d}-c{args.concurrency}"
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    _write_csv(csv_path, measurements)
    summary = _summarize(
        args,
        measurements,
        elapsed,
        metrics_before,
        metrics_after,
    )
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"raw_csv={csv_path}")
    print(f"summary_json={json_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--label", default="cache-enabled")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--repeat-rate", type=float, default=0.4)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--token", default=os.getenv("API_AUTH_TOKEN", ""))
    parser.add_argument(
        "--mode",
        choices=("vector", "keyword", "hybrid"),
        default="hybrid",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--vector-weight", type=float, default=0.7)
    parser.add_argument(
        "--filters",
        type=json.loads,
        default={},
        help='JSON object, for example: {"filenames":["notes.txt"]}',
    )
    parser.add_argument("--baseline-provider-calls", type=int)
    parser.add_argument(
        "--ground-truth-csv",
        type=Path,
        help="CSV from the same post-update trace with caches disabled.",
    )
    parser.add_argument("--embedding-cost-per-call", type=float, default=0.0)
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete embedding and retrieval cache rows before the run.",
    )
    args = parser.parse_args()
    if args.requests <= 0 or args.concurrency <= 0:
        parser.error("--requests and --concurrency must be positive")
    if not 0 <= args.repeat_rate < 1:
        parser.error("--repeat-rate must be in [0, 1)")
    if not 0 <= args.vector_weight <= 1:
        parser.error("--vector-weight must be in [0, 1]")
    return args


def _clear_cache() -> None:
    from sqlalchemy import delete

    from note_rag.persistence import (
        Database,
        QueryEmbeddingCache,
        RetrievalResultCache,
    )

    database = Database()
    try:
        with database.session() as session:
            session.execute(delete(QueryEmbeddingCache))
            session.execute(delete(RetrievalResultCache))
    finally:
        database.dispose()


def _read_queries(path: Path) -> list[str]:
    queries = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not queries:
        raise ValueError("query file contains no queries")
    normalized = {_normalize(query) for query in queries}
    if len(normalized) != len(queries):
        raise ValueError("query file contains duplicates after normalization")
    return queries


def _build_trace(
    queries: list[str],
    *,
    request_count: int,
    repeat_rate: float,
    seed: int,
) -> list[str]:
    duplicate_count = round(request_count * repeat_rate)
    unique_count = request_count - duplicate_count
    if unique_count > len(queries):
        raise ValueError(
            f"workload needs {unique_count} unique queries but file has {len(queries)}"
        )
    rng = random.Random(seed)
    unique = rng.sample(queries, unique_count)
    duplicates = [rng.choice(unique) for _ in range(duplicate_count)]
    trace = [*unique, *duplicates]
    rng.shuffle(trace)
    return trace


def _execute(
    base_url: str,
    query: str,
    index: int,
    headers: dict[str, str],
    timeout: float,
    mode: str,
    top_k: int,
    vector_weight: float,
    filters: dict[str, Any],
) -> Measurement:
    body = json.dumps(
        {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "vector_weight": vector_weight,
            "filters": filters,
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/v1/retrieval/search",
        data=body,
        headers=headers,
        method="POST",
    )
    started = time.perf_counter_ns()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
            status = response.status
            response_headers = response.headers
            error = ""
    except urllib.error.HTTPError as exc:
        payload = {}
        status = exc.code
        response_headers = exc.headers
        error = exc.read().decode(errors="replace")
    except Exception as exc:
        payload = {}
        status = 0
        response_headers = {}
        error = f"{exc.__class__.__name__}: {exc}"
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000
    chunk_ids = ",".join(str(hit["chunk_id"]) for hit in payload.get("hits", []))
    normalized = _normalize(query)
    return Measurement(
        request_index=index,
        query_id=hashlib.sha256(normalized.encode()).hexdigest()[:16],
        normalized_query=normalized,
        latency_ms=round(latency_ms, 3),
        status_code=status,
        embedding_cache_status=response_headers.get("X-Embedding-Cache", "unknown"),
        retrieval_cache_status=response_headers.get("X-Retrieval-Cache", "unknown"),
        corpus_version=response_headers.get("X-Corpus-Version", ""),
        result_chunk_ids=chunk_ids,
        error=error,
    )


def _fetch_metrics(base_url: str, headers: dict[str, str]) -> dict[str, float]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/metrics",
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        lines = response.read().decode().splitlines()
    metrics: dict[str, float] = {}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        name, value = line.rsplit(" ", 1)
        metrics[name] = float(value)
    return metrics


def _summarize(
    args: argparse.Namespace,
    rows: list[Measurement],
    elapsed: float,
    before: dict[str, float],
    after: dict[str, float],
) -> dict[str, Any]:
    successful = [row for row in rows if 200 <= row.status_code < 300]
    cold = [
        row
        for row in successful
        if row.retrieval_cache_status != "hit"
        and row.embedding_cache_status in {"miss", "expired", "invalid"}
    ]
    embedding_warm = [
        row
        for row in successful
        if row.retrieval_cache_status != "hit" and row.embedding_cache_status == "hit"
    ]
    fully_warm = [row for row in successful if row.retrieval_cache_status == "hit"]
    provider_metric = 'note_rag_embedding_provider_calls_total{result="success"}'
    provider_calls = int(after.get(provider_metric, 0) - before.get(provider_metric, 0))
    stale_result_rate = None
    if args.ground_truth_csv is not None:
        expected = _read_ground_truth(args.ground_truth_csv)
        mismatches = sum(
            expected.get(row.request_index) != row.result_chunk_ids
            for row in successful
        )
        stale_result_rate = _ratio(mismatches, len(successful))
    embedding_lookups = sum(
        row.embedding_cache_status in {"hit", "miss", "expired", "invalid"}
        for row in successful
    )
    embedding_hits = sum(row.embedding_cache_status == "hit" for row in successful)
    retrieval_lookups = sum(
        row.retrieval_cache_status in {"hit", "miss", "expired"} for row in successful
    )
    retrieval_hits = sum(row.retrieval_cache_status == "hit" for row in successful)
    api_reduction = None
    if args.baseline_provider_calls:
        api_reduction = (
            args.baseline_provider_calls - provider_calls
        ) / args.baseline_provider_calls
    return {
        "configuration": {
            "label": args.label,
            "requests": args.requests,
            "repeat_rate": args.repeat_rate,
            "concurrency": args.concurrency,
            "seed": args.seed,
            "mode": args.mode,
            "top_k": args.top_k,
            "vector_weight": args.vector_weight,
            "filters": args.filters,
        },
        "completed_requests": len(rows),
        "successful_requests": len(successful),
        "errors": len(rows) - len(successful),
        "elapsed_seconds": round(elapsed, 6),
        "throughput_requests_per_second": round(len(rows) / elapsed, 3),
        "latency_ms": _latencies(successful),
        "cold_latency_ms": _latencies(cold),
        "embedding_warm_latency_ms": _latencies(embedding_warm),
        "fully_warm_latency_ms": _latencies(fully_warm),
        "embedding_cache_hit_rate": _ratio(embedding_hits, embedding_lookups),
        "retrieval_cache_hit_rate": _ratio(retrieval_hits, retrieval_lookups),
        "embedding_provider_calls": provider_calls,
        "embedding_provider_calls_per_query": _ratio(provider_calls, len(successful)),
        "embedding_api_call_reduction": api_reduction,
        "estimated_embedding_cost_per_1000_queries": (
            _ratio(provider_calls, len(successful))
            * 1000
            * args.embedding_cost_per_call
        ),
        "stale_result_rate": stale_result_rate,
        "corpus_versions": sorted(
            {row.corpus_version for row in successful if row.corpus_version}
        ),
    }


def _read_ground_truth(path: Path) -> dict[int, str]:
    with path.open(newline="", encoding="utf-8") as source:
        return {
            int(row["request_index"]): row["result_chunk_ids"]
            for row in csv.DictReader(source)
            if 200 <= int(row["status_code"]) < 300
        }


def _latencies(rows: list[Measurement]) -> dict[str, float | int | None]:
    values = sorted(row.latency_ms for row in rows)
    return {
        "count": len(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
    }


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    position = (len(values) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(values[lower], 3)
    fraction = position - lower
    return round(values[lower] + (values[upper] - values[lower]) * fraction, 3)


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _normalize(query: str) -> str:
    return " ".join(query.strip().split())


def _write_csv(path: Path, rows: list[Measurement]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=Measurement.__dataclass_fields__)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


if __name__ == "__main__":
    main()

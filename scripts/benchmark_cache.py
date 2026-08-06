"""Measure cold/warm retrieval latency and embedding-call reduction."""

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from note_rag.cache import PersistentCache
from note_rag.persistence import Base, ChunkRecord, Database, Document
from note_rag.retrieval import RetrievalService, SearchMode


class DelayedEmbeddingProvider:
    model_name = "benchmark-768"
    dimension = 768

    def __init__(self, delay_ms: float) -> None:
        self.delay_seconds = delay_ms / 1000
        self.calls = 0

    def embed_query(self, query: str) -> list[float]:
        self.calls += 1
        time.sleep(self.delay_seconds)
        return self.vector()

    @staticmethod
    def vector() -> list[float]:
        return [1.0, *([0.0] * 767)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--unique-queries", type=int, default=20)
    parser.add_argument("--embedding-delay-ms", type=float, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.queries <= 0 or not 0 < args.unique_queries <= args.queries:
        parser.error("require 0 < unique-queries <= queries")

    baseline = _run(args, cached=False)
    cached = _run(args, cached=True)
    report = {
        "configuration": {
            "queries": args.queries,
            "unique_queries": args.unique_queries,
            "embedding_delay_ms": args.embedding_delay_ms,
        },
        "baseline": baseline,
        "cached": cached,
        "improvement": {
            "warm_p95_latency_percent": _reduction(
                baseline["p95_latency_ms"], cached["warm_p95_latency_ms"]
            ),
            "embedding_calls_percent": _reduction(
                baseline["embedding_calls"], cached["embedding_calls"]
            ),
            "cache_hit_rate_percent": (
                (args.queries - args.unique_queries) / args.queries * 100
            ),
        },
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


def _run(args: argparse.Namespace, *, cached: bool) -> dict[str, float | int]:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    database = Database(engine=engine)
    with database.session() as session:
        document = Document(filename="benchmark.txt", media_type="text/plain")
        session.add(document)
        session.add(
            ChunkRecord(
                document=document,
                position=0,
                text="cache benchmark content",
                token_count=3,
                token_start=0,
                token_end=3,
                char_start=0,
                char_end=23,
                source_metadata={},
                embedding=DelayedEmbeddingProvider.vector(),
            )
        )
    provider = DelayedEmbeddingProvider(args.embedding_delay_ms)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        cache = (
            PersistentCache(Path(directory) / "cache.sqlite3")
            if cached
            else None
        )
        service = RetrievalService(database, provider, cache=cache)
        latencies: list[float] = []
        warm_latencies: list[float] = []
        for index in range(args.queries):
            query = f"benchmark query {index % args.unique_queries}"
            started = time.perf_counter()
            service.search(query, mode=SearchMode.VECTOR)
            elapsed = (time.perf_counter() - started) * 1000
            latencies.append(elapsed)
            if cached and index >= args.unique_queries:
                warm_latencies.append(elapsed)
    database.dispose()
    ordered = sorted(latencies)
    p95_index = max(0, int(len(ordered) * 0.95) - 1)
    result: dict[str, float | int] = {
        "mean_latency_ms": statistics.fmean(latencies),
        "p95_latency_ms": ordered[p95_index],
        "embedding_calls": provider.calls,
    }
    if warm_latencies:
        ordered_warm = sorted(warm_latencies)
        warm_p95_index = max(0, int(len(ordered_warm) * 0.95) - 1)
        result["warm_mean_latency_ms"] = statistics.fmean(warm_latencies)
        result["warm_p95_latency_ms"] = ordered_warm[warm_p95_index]
    else:
        result["warm_mean_latency_ms"] = result["mean_latency_ms"]
        result["warm_p95_latency_ms"] = result["p95_latency_ms"]
    return result


def _reduction(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline * 100 if baseline else 0.0


if __name__ == "__main__":
    main()

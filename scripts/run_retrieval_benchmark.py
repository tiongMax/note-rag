"""Run the gold retrieval dataset against one isolated Note RAG index."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from note_rag.context import (  # noqa: E402
    CrossEncoderReranker,
    LexicalReranker,
    Reranker,
)
from note_rag.evaluation import (  # noqa: E402
    Bm25Index,
    GoldPassage,
    RetrievedChunk,
    aggregate_query_metrics,
    evaluate_query,
    passage_coverage,
    weighted_rrf,
)

_SAFE_LABEL = re.compile(r"^[A-Za-z0-9._-]+$")


class ApiClient:
    def __init__(self, base_url: str, token: str = "", timeout: float = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {"Accept": "application/json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        body = None
        headers = dict(self.headers)
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"{method} {path} failed with HTTP {error.code}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"could not reach benchmark API at {self.base_url}: {error.reason}"
            ) from error


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    entries = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        entries.append(value)
    if not entries:
        raise ValueError("gold dataset is empty")
    return entries


def _verify_corpus(
    client: ApiClient,
    expected_filenames: set[str],
) -> list[dict[str, Any]]:
    documents = client.request("GET", "/api/v1/documents")
    if not isinstance(documents, list):
        raise RuntimeError("documents endpoint returned an unexpected response")
    by_filename = {document.get("filename"): document for document in documents}
    missing = sorted(expected_filenames - by_filename.keys())
    if missing:
        raise RuntimeError(
            "benchmark documents are missing from the index: " + ", ".join(missing)
        )
    unavailable = []
    for filename in sorted(expected_filenames):
        document = by_filename[filename]
        if (
            document.get("status") != "ready"
            or document.get("indexing_status") != "indexed"
        ):
            unavailable.append(
                f"{filename} (status={document.get('status')}, "
                f"indexing_status={document.get('indexing_status')})"
            )
    if unavailable:
        raise RuntimeError(
            "benchmark documents are not ready and indexed: " + "; ".join(unavailable)
        )
    return [by_filename[filename] for filename in sorted(expected_filenames)]


def _index_statistics(
    client: ApiClient,
    documents: list[dict[str, Any]],
) -> dict[str, float | int]:
    token_counts = []
    for document in documents:
        chunks = client.request(
            "GET",
            f"/api/v1/documents/{document['id']}/chunks",
        )
        if not isinstance(chunks, list):
            raise RuntimeError(
                f"{document['filename']}: chunks endpoint returned "
                "an unexpected response"
            )
        token_counts.extend(int(chunk["token_count"]) for chunk in chunks)
    if not token_counts:
        raise RuntimeError("benchmark index contains no chunks")
    ordered = sorted(token_counts)
    p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "embedding_budget_tokens": sum(token_counts),
        "chunk_tokens_mean": statistics.fmean(token_counts),
        "chunk_tokens_p50": statistics.median(token_counts),
        "chunk_tokens_p95": ordered[p95_index],
        "chunk_tokens_min": ordered[0],
        "chunk_tokens_max": ordered[-1],
    }


def _bm25_corpus(
    client: ApiClient,
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    corpus = []
    for document in documents:
        chunks = client.request(
            "GET",
            f"/api/v1/documents/{document['id']}/chunks",
        )
        for chunk in chunks:
            corpus.append(
                {
                    **chunk,
                    "chunk_id": chunk["id"],
                    "document_id": document["id"],
                    "filename": document["filename"],
                    "media_type": document.get(
                        "media_type",
                        "application/octet-stream",
                    ),
                    "vector_score": None,
                    "keyword_score": None,
                }
            )
    return corpus


def _api_search(
    client: ApiClient,
    *,
    query: str,
    mode: str,
    limit: int,
    vector_weight: float,
    filenames: list[str],
) -> list[dict[str, Any]]:
    response = client.request(
        "POST",
        "/api/v1/retrieval/search",
        {
            "query": query,
            "mode": mode,
            "top_k": limit,
            "vector_weight": vector_weight,
            "filters": {"filenames": filenames},
        },
    )
    hits = response.get("hits")
    if not isinstance(hits, list):
        raise RuntimeError("search response has no hit list")
    return hits


def _retrieve(
    *,
    client: ApiClient,
    query: str,
    system: str,
    limit: int,
    vector_weight: float,
    filenames: list[str],
    bm25_index: Bm25Index | None,
    rrf_k: int,
) -> list[dict[str, Any]]:
    if system in {"vector", "keyword", "current_hybrid"}:
        mode = "hybrid" if system == "current_hybrid" else system
        return _api_search(
            client,
            query=query,
            mode=mode,
            limit=limit,
            vector_weight=vector_weight,
            filenames=filenames,
        )
    if bm25_index is None:
        raise RuntimeError("BM25 system requested without a BM25 index")
    bm25_hits = bm25_index.search(query, limit=limit)
    if system == "bm25":
        return bm25_hits
    dense_hits = _api_search(
        client,
        query=query,
        mode="vector",
        limit=limit,
        vector_weight=1.0,
        filenames=filenames,
    )
    return weighted_rrf(
        (
            (dense_hits, vector_weight),
            (bm25_hits, 1.0 - vector_weight),
        ),
        rrf_k=rrf_k,
    )


def _rerank(
    query: str,
    hits: list[dict[str, Any]],
    reranker: Reranker | None,
) -> list[dict[str, Any]]:
    if reranker is None or not hits:
        return hits
    scores = reranker.score(query, [str(hit["text"]) for hit in hits])
    ranked = [
        {**hit, "rerank_score": score, "_original_rank": rank}
        for rank, (hit, score) in enumerate(zip(hits, scores, strict=True))
    ]
    ranked.sort(
        key=lambda hit: (
            -float(hit["rerank_score"]),
            int(hit["_original_rank"]),
        )
    )
    for hit in ranked:
        hit.pop("_original_rank")
    return ranked


def _gold_passages(entry: dict[str, Any]) -> list[GoldPassage]:
    return [
        GoldPassage(
            source_id=passage["source_id"],
            char_start=passage["char_start"],
            char_end=passage["char_end"],
        )
        for passage in entry["relevant_passages"]
    ]


def _retrieved_chunks(hits: list[dict[str, Any]]) -> list[RetrievedChunk]:
    chunks = []
    for rank, hit in enumerate(hits, start=1):
        missing = [
            field
            for field in ("filename", "char_start", "char_end")
            if field not in hit
        ]
        if missing:
            raise RuntimeError(
                f"retrieval hit at rank {rank} is missing {missing}; "
                "restart the API with the benchmark-compatible search response"
            )
        chunks.append(
            RetrievedChunk(
                source_id=hit["filename"],
                char_start=hit["char_start"],
                char_end=hit["char_end"],
            )
        )
    return chunks


def _enriched_hits(
    hits: list[dict[str, Any]],
    chunks: list[RetrievedChunk],
    passages: list[GoldPassage],
    relevance_threshold: float,
) -> list[dict[str, Any]]:
    enriched = []
    for rank, (hit, chunk) in enumerate(zip(hits, chunks, strict=True), start=1):
        coverages = [
            passage_coverage(chunk, passage) for passage in passages
        ]
        enriched.append(
            {
                **hit,
                "rank": rank,
                "gold_passage_coverages": coverages,
                "max_gold_passage_coverage": max(coverages),
                "relevant": max(coverages) >= relevance_threshold,
            }
        )
    return enriched


def _ranking_signature(hits: list[dict[str, Any]]) -> list[str]:
    return [str(hit["chunk_id"]) for hit in hits]


def _latency_statistics(
    values: list[float],
    prefix: str,
) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {
            f"{prefix}_mean_ms": 0.0,
            f"{prefix}_p50_ms": 0.0,
            f"{prefix}_p95_ms": 0.0,
        }

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * weight

    return {
        f"{prefix}_mean_ms": statistics.fmean(ordered),
        f"{prefix}_p50_ms": percentile(0.50),
        f"{prefix}_p95_ms": percentile(0.95),
    }


def run_benchmark(
    *,
    client: ApiClient,
    entries: list[dict[str, Any]],
    label: str,
    mode: str,
    top_k: int,
    vector_weight: float,
    relevance_threshold: float,
    repetitions: int,
    delay_ms: float,
    system: str | None = None,
    candidate_k: int | None = None,
    reranker: Reranker | None = None,
    rrf_k: int = 60,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> tuple[
    list[dict[str, Any]],
    dict[str, float | int],
    list[dict[str, Any]],
]:
    filenames = {
        passage["source_id"]
        for entry in entries
        for passage in entry["relevant_passages"]
    }
    corpus_documents = _verify_corpus(client, filenames)
    index_statistics = _index_statistics(client, corpus_documents)
    resolved_system = system or (
        "current_hybrid" if mode == "hybrid" else mode
    )
    resolved_candidate_k = candidate_k or top_k
    bm25_started = time.perf_counter()
    bm25_index = (
        Bm25Index(
            _bm25_corpus(client, corpus_documents),
            k1=bm25_k1,
            b=bm25_b,
        )
        if resolved_system in {"bm25", "bm25_dense"}
        else None
    )
    bm25_build_ms = (time.perf_counter() - bm25_started) * 1000

    query_results = []
    all_latencies_ms: list[float] = []
    retrieval_latencies_ms: list[float] = []
    rerank_latencies_ms: list[float] = []
    for number, entry in enumerate(entries, start=1):
        responses = []
        latencies_ms = []
        query_retrieval_latencies_ms = []
        query_rerank_latencies_ms = []
        for repetition in range(repetitions):
            retrieval_started = time.perf_counter()
            candidates = _retrieve(
                client=client,
                query=entry["query"],
                system=resolved_system,
                limit=resolved_candidate_k,
                vector_weight=vector_weight,
                filenames=sorted(filenames),
                bm25_index=bm25_index,
                rrf_k=rrf_k,
            )
            retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
            rerank_started = time.perf_counter()
            hits = _rerank(entry["query"], candidates, reranker)[:top_k]
            rerank_ms = (time.perf_counter() - rerank_started) * 1000
            latency_ms = retrieval_ms + rerank_ms
            responses.append(hits)
            latencies_ms.append(latency_ms)
            all_latencies_ms.append(latency_ms)
            query_retrieval_latencies_ms.append(retrieval_ms)
            query_rerank_latencies_ms.append(rerank_ms)
            retrieval_latencies_ms.append(retrieval_ms)
            rerank_latencies_ms.append(rerank_ms)
            if delay_ms and (repetition < repetitions - 1 or number < len(entries)):
                time.sleep(delay_ms / 1000)

        hits = responses[0]
        passages = _gold_passages(entry)
        chunks = _retrieved_chunks(hits)
        metrics = evaluate_query(
            passages,
            chunks,
            relevance_threshold=relevance_threshold,
        )
        signatures = [_ranking_signature(response) for response in responses]
        query_results.append(
            {
                "schema_version": "1.0",
                "run_label": label,
                "id": entry["id"],
                "query": entry["query"],
                "expected_answer": entry["expected_answer"],
                "tags": entry["tags"],
                "gold_passages": entry["relevant_passages"],
                "metrics": metrics,
                "latencies_ms": latencies_ms,
                "retrieval_latencies_ms": query_retrieval_latencies_ms,
                "rerank_latencies_ms": query_rerank_latencies_ms,
                "ranking_stable": all(
                    signature == signatures[0] for signature in signatures[1:]
                ),
                "hits": _enriched_hits(
                    hits,
                    chunks,
                    passages,
                    relevance_threshold,
                ),
            }
        )
        print(
            f"[{number:02}/{len(entries)}] {entry['id']}: "
            f"R@10={metrics['recall_at_10']:.3f}, "
            f"MRR={metrics['mrr']:.3f}, "
            f"{latencies_ms[0]:.1f} ms"
        )

    summary = aggregate_query_metrics(
        [result["metrics"] for result in query_results],
        all_latencies_ms,
    )
    summary["unstable_ranking_count"] = sum(
        not result["ranking_stable"] for result in query_results
    )
    summary["document_count"] = len(corpus_documents)
    summary["chunk_count"] = sum(
        int(document["chunk_count"]) for document in corpus_documents
    )
    summary["document_token_count"] = sum(
        int(document["token_count"]) for document in corpus_documents
    )
    summary.update(
        _latency_statistics(retrieval_latencies_ms, "retrieval_latency")
    )
    summary.update(_latency_statistics(rerank_latencies_ms, "rerank_latency"))
    summary["queries_per_second"] = (
        len(all_latencies_ms) / (sum(all_latencies_ms) / 1000)
        if sum(all_latencies_ms)
        else 0.0
    )
    summary["bm25_build_ms"] = bm25_build_ms
    summary["bm25_document_count"] = (
        bm25_index.document_count if bm25_index is not None else 0
    )
    summary.update(index_statistics)
    return query_results, summary, corpus_documents


def _write_results(
    output_dir: Path,
    label: str,
    query_results: list[dict[str, Any]],
    metadata: dict[str, Any],
    summary: dict[str, float | int],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{label}.results.jsonl"
    summary_path = output_dir / f"{label}.summary.json"
    csv_path = output_dir / f"{label}.summary.csv"

    results_path.write_text(
        "".join(
            json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"
            for result in query_results
        ),
        encoding="utf-8",
        newline="\n",
    )
    summary_document = {"metadata": metadata, "metrics": summary}
    summary_path.write_text(
        json.dumps(summary_document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    csv_row = {**metadata, **summary}
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_row.keys())
        writer.writeheader()
        writer.writerow(csv_row)
    return results_path, summary_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPOSITORY_ROOT / "benchmarks/gold/dataset.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "tmp/benchmarks",
    )
    parser.add_argument(
        "--mode",
        choices=("vector", "keyword", "hybrid"),
        default="hybrid",
    )
    parser.add_argument(
        "--system",
        choices=("vector", "keyword", "current_hybrid", "bm25", "bm25_dense"),
        help="retrieval system; defaults to the legacy --mode selection",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--vector-weight", type=float, default=0.7)
    parser.add_argument("--relevance-threshold", type=float, default=0.5)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--reranker",
        choices=("none", "lexical", "cross_encoder"),
        default="none",
    )
    parser.add_argument(
        "--cross-encoder-model",
        default="cross-encoder/ms-marco-MiniLM-L6-v2",
    )
    parser.add_argument("--cross-encoder-device", default="")
    parser.add_argument("--cross-encoder-batch-size", type=int, default=16)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--delay-ms", type=float, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--api-token-env", default="API_AUTH_TOKEN")
    args = parser.parse_args()

    if not _SAFE_LABEL.fullmatch(args.label):
        parser.error("--label may contain only letters, digits, '.', '_', and '-'")
    if args.top_k < 20:
        parser.error("--top-k must be at least 20 to calculate @20 metrics")
    if args.candidate_k < args.top_k:
        parser.error("--candidate-k must be at least --top-k")
    if args.candidate_k > 100:
        parser.error("--candidate-k cannot exceed the API limit of 100")
    if not 0 <= args.vector_weight <= 1:
        parser.error("--vector-weight must be between 0 and 1")
    if args.rrf_k < 0:
        parser.error("--rrf-k cannot be negative")
    if args.bm25_k1 <= 0:
        parser.error("--bm25-k1 must be greater than zero")
    if not 0 <= args.bm25_b <= 1:
        parser.error("--bm25-b must be between 0 and 1")
    if args.cross_encoder_batch_size <= 0:
        parser.error("--cross-encoder-batch-size must be greater than zero")
    if not 0 < args.relevance_threshold <= 1:
        parser.error("--relevance-threshold must be in (0, 1]")
    if args.repetitions <= 0:
        parser.error("--repetitions must be greater than zero")
    if args.delay_ms < 0:
        parser.error("--delay-ms cannot be negative")

    dataset_path = args.dataset.resolve()
    entries = _load_dataset(dataset_path)
    dataset_bytes = dataset_path.read_bytes()
    token = os.getenv(args.api_token_env, "") if args.api_token_env else ""
    client = ApiClient(
        args.base_url,
        token=token,
        timeout=args.timeout_seconds,
    )
    reranker: Reranker | None
    if args.reranker == "lexical":
        reranker = LexicalReranker()
    elif args.reranker == "cross_encoder":
        reranker = CrossEncoderReranker(
            args.cross_encoder_model,
            device=args.cross_encoder_device or None,
            batch_size=args.cross_encoder_batch_size,
        )
    else:
        reranker = None
    query_results, summary, corpus_documents = run_benchmark(
        client=client,
        entries=entries,
        label=args.label,
        mode=args.mode,
        top_k=args.top_k,
        vector_weight=args.vector_weight,
        relevance_threshold=args.relevance_threshold,
        repetitions=args.repetitions,
        delay_ms=args.delay_ms,
        system=args.system,
        candidate_k=args.candidate_k if args.reranker != "none" else args.top_k,
        reranker=reranker,
        rrf_k=args.rrf_k,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
    )
    metadata = {
        "schema_version": "1.0",
        "run_label": args.label,
        "created_at": datetime.now(UTC).isoformat(),
        "base_url": args.base_url.rstrip("/"),
        "dataset": dataset_path.as_posix(),
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "mode": args.mode,
        "system": args.system or (
            "current_hybrid" if args.mode == "hybrid" else args.mode
        ),
        "top_k": args.top_k,
        "candidate_k": (
            args.candidate_k if args.reranker != "none" else args.top_k
        ),
        "vector_weight": args.vector_weight,
        "reranker": args.reranker,
        "reranker_model": reranker.model_name if reranker is not None else None,
        "rrf_k": args.rrf_k,
        "bm25_k1": args.bm25_k1,
        "bm25_b": args.bm25_b,
        "relevance_threshold": args.relevance_threshold,
        "repetitions": args.repetitions,
        "embedding_models": ",".join(
            sorted(
                {
                    str(document["embedding_model"])
                    for document in corpus_documents
                }
            )
        ),
    }
    paths = _write_results(
        args.output_dir.resolve(),
        args.label,
        query_results,
        metadata,
        summary,
    )
    print(json.dumps(summary, indent=2))
    print("Wrote:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()

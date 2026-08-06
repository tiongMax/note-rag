from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_script() -> ModuleType:
    path = (
        Path(__file__).parents[2]
        / "scripts"
        / "run_retrieval_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeClient:
    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        if method == "GET" and path == "/api/v1/documents":
            return [
                {
                    "id": "document-1",
                    "filename": "source.pdf",
                    "status": "ready",
                    "indexing_status": "indexed",
                    "embedding_model": "fake-model",
                    "chunk_count": 10,
                    "token_count": 1000,
                }
            ]
        if method == "GET":
            assert path == "/api/v1/documents/document-1/chunks"
            return [
                {
                    "id": f"chunk-{rank}",
                    "document_id": "document-1",
                    "position": rank - 1,
                    "text": (
                        "support exact marker"
                        if rank == 1
                        else f"unrelated text {rank}"
                    ),
                    "token_count": 100,
                    "token_start": (rank - 1) * 100,
                    "token_end": rank * 100,
                    "char_start": 100 if rank == 1 else rank * 1000,
                    "char_end": 200 if rank == 1 else rank * 1000 + 100,
                    "source_metadata": {},
                }
                for rank in range(1, 11)
            ]
        assert path == "/api/v1/retrieval/search"
        assert payload is not None
        assert payload["filters"] == {"filenames": ["source.pdf"]}
        return {
            "hits": [
                {
                    "chunk_id": f"chunk-{rank}",
                    "filename": "source.pdf",
                    "char_start": 100 if rank == 1 else rank * 1000,
                    "char_end": 200 if rank == 1 else rank * 1000 + 100,
                    "score": 1 / rank,
                }
                for rank in range(1, 11)
            ]
        }


def test_runs_one_query_and_records_system_counts() -> None:
    runner = _load_script()
    entries = [
        {
            "id": "q1",
            "query": "question",
            "expected_answer": "answer",
            "tags": ["test"],
            "relevant_passages": [
                {
                    "source_id": "source.pdf",
                    "source_path": "documents/source.pdf",
                    "page_number": 1,
                    "char_start": 100,
                    "char_end": 200,
                    "reference_text": "support",
                    "text_sha256": "unused",
                }
            ],
        }
    ]

    results, summary, documents = runner.run_benchmark(
        client=FakeClient(),
        entries=entries,
        label="fixed",
        mode="hybrid",
        top_k=10,
        vector_weight=0.7,
        relevance_threshold=0.5,
        repetitions=1,
        delay_ms=0,
    )

    assert results[0]["metrics"]["recall_at_10"] == 1
    assert results[0]["hits"][0]["relevant"] is True
    assert summary["chunk_count"] == 10
    assert summary["document_token_count"] == 1000
    assert summary["embedding_budget_tokens"] == 1000
    assert summary["chunk_tokens_p95"] == 100
    assert documents[0]["embedding_model"] == "fake-model"


def test_runs_in_process_bm25_against_the_same_gold_passages() -> None:
    runner = _load_script()
    entries = [
        {
            "id": "q1",
            "query": "support exact marker",
            "expected_answer": "answer",
            "tags": ["exact-term"],
            "relevant_passages": [
                {
                    "source_id": "source.pdf",
                    "char_start": 100,
                    "char_end": 200,
                }
            ],
        }
    ]

    results, summary, _ = runner.run_benchmark(
        client=FakeClient(),
        entries=entries,
        label="bm25",
        mode="hybrid",
        system="bm25",
        top_k=10,
        candidate_k=10,
        vector_weight=0.7,
        relevance_threshold=0.5,
        repetitions=1,
        delay_ms=0,
    )

    assert results[0]["hits"][0]["chunk_id"] == "chunk-1"
    assert results[0]["metrics"]["mrr"] == 1
    assert summary["bm25_document_count"] == 10

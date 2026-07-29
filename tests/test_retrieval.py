"""Unit tests for retrieval strategy behavior."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from langchain_core.documents import Document

from note_rag.retrieval import (
    RetrievalConfig,
    RetrievalPipeline,
    RetrievalStrategy,
    build_bm25_retriever,
    document_chunk_id,
    reciprocal_rank_fusion,
)


def doc(chunk_id: str, text: str = "text") -> Document:
    return Document(id=chunk_id, page_content=text)


class FakeVectorStore:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.requested_k: list[int] = []

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 4
    ) -> list[tuple[Document, float]]:
        self.requested_k.append(k)
        return [
            (document, 1.0 - index / 100)
            for index, document in enumerate(self.documents[:k])
        ]


class RecordingReranker:
    def __init__(self) -> None:
        self.candidate_count = 0

    def rerank(self, query: str, documents: list[Document], k: int) -> list[Document]:
        self.candidate_count = len(documents)
        return list(reversed(documents))[:k]


class RetrievalTests(unittest.TestCase):
    def test_chunk_id_prefers_metadata_then_document_id(self) -> None:
        document = Document(
            id="database-id",
            page_content="content",
            metadata={"chunk_id": "metadata-id"},
        )
        self.assertEqual(document_chunk_id(document), "metadata-id")

    def test_rrf_deduplicates_and_rewards_overlap(self) -> None:
        fused = reciprocal_rank_fusion(
            [[doc("a"), doc("b")], [doc("b"), doc("c")]],
            [0.5, 0.5],
            limit=3,
        )
        ids = [document_chunk_id(value) for value in fused]
        self.assertEqual(ids, ["b", "a", "c"])

    def test_dense_only_requests_five(self) -> None:
        store = FakeVectorStore([doc(str(index)) for index in range(10)])
        pipeline = RetrievalPipeline(
            store,
            RetrievalConfig(strategy=RetrievalStrategy.DENSE_ONLY),
        )
        self.assertEqual(len(pipeline.retrieve("question")), 5)
        self.assertEqual(store.requested_k, [5])

    def test_hybrid_rerank_scores_twenty_and_returns_five(self) -> None:
        documents = [doc(str(index), f"shared token {index}") for index in range(30)]
        store = FakeVectorStore(documents)
        reranker = RecordingReranker()
        pipeline = RetrievalPipeline(
            store,
            RetrievalConfig(strategy=RetrievalStrategy.HYBRID_RERANK),
            bm25_retriever=build_bm25_retriever(documents),
            reranker=reranker,  # type: ignore[arg-type]
            verbose=True,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            results = pipeline.retrieve("shared")
        self.assertEqual(store.requested_k, [20])
        self.assertEqual(reranker.candidate_count, 20)
        self.assertEqual(len(results), 5)
        self.assertIn("BM25 candidates (20)", output.getvalue())
        self.assertIn("Cross-encoder reranked top 5 (5)", output.getvalue())
        self.assertIn("Dense vector candidates (20)", output.getvalue())
        self.assertIn("similarity=1.0000", output.getvalue())


if __name__ == "__main__":
    unittest.main()

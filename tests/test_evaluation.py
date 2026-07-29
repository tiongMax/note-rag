"""Unit tests for the ranked retrieval and end-to-end evaluation harness."""

from __future__ import annotations

import math
import unittest

from langchain_core.documents import Document

from note_rag.evaluation import (
    EvalCase,
    evaluate_generation,
    evaluate_retrievers,
    precision_recall_at_5,
    ranking_metrics_at_k,
)


class FixedPipeline:
    def __init__(self, ids: str = "abcde") -> None:
        self.ids = ids

    def retrieve(self, query: str) -> list[Document]:
        return [
            Document(id=value, page_content=f"context {value}") for value in self.ids
        ]


class EvaluationTests(unittest.TestCase):
    def test_standard_ranking_metrics_reward_early_relevant_results(self) -> None:
        metrics = ranking_metrics_at_k(
            ["a", "x", "b", "y", "z"],
            ["a", "b", "missing"],
        )

        self.assertEqual(metrics["precision_at_k"], 0.4)
        self.assertAlmostEqual(metrics["recall_at_k"], 2 / 3)
        self.assertAlmostEqual(metrics["f1_at_k"], 0.5)
        self.assertEqual(metrics["hit_rate_at_k"], 1.0)
        self.assertEqual(metrics["reciprocal_rank"], 1.0)
        self.assertAlmostEqual(
            metrics["average_precision_at_k"],
            (1.0 + 2 / 3) / 3,
        )
        ideal_dcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
        self.assertAlmostEqual(
            metrics["ndcg_at_k"],
            (1 + 1 / math.log2(4)) / ideal_dcg,
        )

    def test_reciprocal_rank_penalizes_a_late_first_hit(self) -> None:
        metrics = ranking_metrics_at_k(["x", "y", "a", "z", "q"], ["a"])
        self.assertAlmostEqual(metrics["reciprocal_rank"], 1 / 3)
        self.assertAlmostEqual(metrics["average_precision_at_k"], 1 / 3)

    def test_precision_and_recall_compatibility_helper(self) -> None:
        precision, recall = precision_recall_at_5(
            ["a", "b", "c", "d", "e"], ["a", "c", "missing"]
        )
        self.assertEqual(precision, 0.4)
        self.assertAlmostEqual(recall, 2 / 3)

    def test_out_of_corpus_is_excluded_from_ir_averages(self) -> None:
        cases: list[EvalCase] = [
            {
                "id": "in",
                "question": "in corpus",
                "answer": "a",
                "relevant_chunk_ids": ["a", "c"],
                "category": "multi-chunk",
            },
            {
                "id": "out",
                "question": "out of corpus",
                "answer": "not covered",
                "relevant_chunk_ids": [],
                "category": "out-of-corpus",
            },
        ]

        details, summaries = evaluate_retrievers(
            cases,
            {"dense-only": FixedPipeline()},  # type: ignore[dict-item]
        )

        self.assertEqual(details[0]["returned_chunk_ids"], list("abcde"))
        self.assertEqual(details[0]["relevant_ranks"], [1, 3])
        self.assertIsNone(details[1]["precision_at_k"])
        in_corpus = next(row for row in summaries if row["slice"] == "in-corpus")
        self.assertEqual(in_corpus["query_count"], 1)
        self.assertEqual(in_corpus["precision_at_k"], 0.4)
        self.assertEqual(in_corpus["recall_at_k"], 1.0)

    def test_generation_evaluation_separates_refusal_slice(self) -> None:
        cases: list[EvalCase] = [
            {
                "id": "in",
                "question": "known",
                "answer": "reference",
                "relevant_chunk_ids": ["a"],
                "category": "single-chunk",
            },
            {
                "id": "out",
                "question": "unknown",
                "answer": "not covered",
                "relevant_chunk_ids": [],
                "category": "out-of-corpus",
            },
        ]

        def answer_generator(question: str, documents: list[Document]) -> str:
            return f"answer to {question}"

        def judge(**kwargs: object) -> dict[str, object]:
            is_out = kwargs["category"] == "out-of-corpus"
            return {
                "correctness": 0.8,
                "faithfulness": 0.9,
                "answer_relevance": 0.7,
                "context_relevance": 0.6,
                "refusal_correctness": 1.0 if is_out else None,
                "explanation": "test",
            }

        details, summaries = evaluate_generation(
            cases,
            config_name="hybrid",
            pipeline=FixedPipeline(),  # type: ignore[arg-type]
            answer_generator=answer_generator,  # type: ignore[arg-type]
            judge=judge,  # type: ignore[arg-type]
        )

        self.assertEqual(len(details), 2)
        out_summary = next(row for row in summaries if row["slice"] == "out-of-corpus")
        self.assertEqual(out_summary["refusal_correctness"], 1.0)


if __name__ == "__main__":
    unittest.main()

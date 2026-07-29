"""End-to-end answer generation evaluation."""

from statistics import fmean

from note_rag.evaluation.models import (
    AnswerGenerator,
    EvalCase,
    GenerationResult,
    GenerationSummary,
    RagJudge,
)
from note_rag.retrieval import RetrievalPipeline, document_chunk_id


def evaluate_generation(
    cases: list[EvalCase],
    *,
    config_name: str,
    pipeline: RetrievalPipeline,
    answer_generator: AnswerGenerator,
    judge: RagJudge,
) -> tuple[list[GenerationResult], list[GenerationSummary]]:
    """Evaluate answer quality, grounding, relevance, and refusal behavior."""

    results: list[GenerationResult] = []
    for index, case in enumerate(cases, start=1):
        print(f"  [{index}/{len(cases)}] generating and judging {case.get('id', '')}")
        documents = pipeline.retrieve(case["question"])
        generated_answer = answer_generator(case["question"], documents)
        scores = judge(
            question=case["question"],
            reference_answer=case["answer"],
            generated_answer=generated_answer,
            contexts=[document.page_content for document in documents],
            category=case["category"],
        )
        results.append(
            {
                "case_id": case.get("id", f"q{index:03d}"),
                "config": config_name,
                "question": case["question"],
                "category": case["category"],
                "reference_answer": case["answer"],
                "generated_answer": generated_answer,
                "returned_chunk_ids": [
                    document_chunk_id(document) for document in documents
                ],
                **scores,
            }
        )
    return results, _summarize(config_name, results)


def _summarize(
    config_name: str, results: list[GenerationResult]
) -> list[GenerationSummary]:
    slices = {
        "in-corpus": [
            result for result in results if result["category"] != "out-of-corpus"
        ],
        "out-of-corpus": [
            result for result in results if result["category"] == "out-of-corpus"
        ],
    }
    summaries: list[GenerationSummary] = []
    for slice_name, values in slices.items():
        if not values:
            continue
        refusals = [
            score
            for result in values
            if (score := result["refusal_correctness"]) is not None
        ]
        summaries.append(
            {
                "config": config_name,
                "slice": slice_name,
                "query_count": len(values),
                "correctness": fmean(result["correctness"] for result in values),
                "faithfulness": fmean(result["faithfulness"] for result in values),
                "answer_relevance": fmean(
                    result["answer_relevance"] for result in values
                ),
                "context_relevance": fmean(
                    result["context_relevance"] for result in values
                ),
                "refusal_correctness": fmean(refusals) if refusals else None,
            }
        )
    return summaries

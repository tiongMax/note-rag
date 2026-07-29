"""Structured LLM judging for RAG evaluation."""

import json
from collections.abc import Sequence
from typing import cast

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from note_rag.config import Settings
from note_rag.evaluation.models import GenerationScores, RagJudge


class RagJudgeGrade(BaseModel):
    """Structured scores returned by the Gemini RAG judge."""

    correctness: float = Field(ge=0.0, le=1.0)
    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevance: float = Field(ge=0.0, le=1.0)
    context_relevance: float = Field(ge=0.0, le=1.0)
    refusal_correctness: float | None = Field(default=None, ge=0.0, le=1.0)
    explanation: str


def build_rag_judge(settings: Settings) -> RagJudge:
    """Build a structured LLM judge for end-to-end RAG evaluation."""

    instructions = """You are an impartial evaluator of a RAG system.
Score each dimension from 0.0 to 1.0 using the supplied evidence:
- correctness: factual agreement and completeness versus the reference answer.
- faithfulness: every factual claim in the generated answer is supported by context.
- answer_relevance: the generated answer directly addresses the question.
- context_relevance: the retrieved context is useful for answering the question.
- refusal_correctness: for out-of-corpus cases, score whether the answer correctly
  refuses or states that the context does not contain the answer. Return null for
  in-corpus cases.
Do not reward unsupported details even when they happen to match the reference.
Use intermediate scores for partially satisfied criteria. Provide only a concise
evidence-based explanation, not hidden reasoning."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", instructions), ("human", "Evaluate this JSON payload:\n{payload}")]
    )
    chain = prompt | ChatGoogleGenerativeAI(
        model=settings.eval_judge_model,
        api_key=settings.google_api_key,
        temperature=0.0,
        max_retries=2,
    ).with_structured_output(RagJudgeGrade)

    def judge(
        *,
        question: str,
        reference_answer: str,
        generated_answer: str,
        contexts: Sequence[str],
        category: str,
    ) -> GenerationScores:
        payload = json.dumps(
            {
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                "retrieved_contexts": list(contexts),
                "category": category,
            },
            ensure_ascii=False,
        )
        grade = cast("RagJudgeGrade", chain.invoke({"payload": payload}))
        return {
            "correctness": grade.correctness,
            "faithfulness": grade.faithfulness,
            "answer_relevance": grade.answer_relevance,
            "context_relevance": grade.context_relevance,
            "refusal_correctness": grade.refusal_correctness,
            "explanation": grade.explanation,
        }

    return judge

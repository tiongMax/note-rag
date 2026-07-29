"""RAG answer generation and LLM-based evaluation judging."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict, cast

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_google_genai import ChatGoogleGenerativeAI

from note_rag.config import Settings
from note_rag.evaluation import AnswerGenerator
from note_rag.rag.judge import RagJudgeGrade, build_rag_judge


class RagResult(TypedDict):
    """Values returned after retrieval and answer generation."""

    question: str
    source_documents: list[Document]
    answer: str


def documents_to_context(documents: Sequence[Document]) -> str:
    """Join retrieved documents into the context format used by generation."""

    return "\n\n--- Retrieved chunk ---\n\n".join(
        document.page_content for document in documents
    )


def answer_prompt() -> ChatPromptTemplate:
    """Create the production answer prompt."""

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's question using the retrieved PDF context."
                "\n\nContext:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )


def _print_retrieved_chunks(documents: list[Document]) -> list[Document]:
    print(f"\nRetrieved chunks ({len(documents)}):")
    for position, document in enumerate(documents, start=1):
        print(f"\n--- Chunk {position} ---")
        print(f"Metadata: {document.metadata}")
        print(" ".join(document.page_content.split()))
    return documents


def _format_prompt_values(inputs: dict[str, Any]) -> dict[str, str]:
    documents = cast("list[Document]", inputs["source_documents"])
    return {
        "question": cast("str", inputs["question"]),
        "context": documents_to_context(documents),
    }


def build_rag_chain(
    retriever: Runnable[str, list[Document]], settings: Settings
) -> Runnable[str, RagResult]:
    """Build retrieval and Gemini answer generation with LCEL."""

    retrieval = retriever | RunnableLambda(_print_retrieved_chunks)
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        api_key=settings.google_api_key,
        temperature=1.0,
        max_retries=2,
    )
    generation = (
        RunnableLambda(_format_prompt_values)
        | answer_prompt()
        | llm
        | StrOutputParser()
    )
    chain = RunnableParallel(
        question=RunnablePassthrough(),
        source_documents=retrieval,
    ) | RunnablePassthrough.assign(answer=generation)
    return cast("Runnable[str, RagResult]", chain)


def build_evaluation_answer_generator(settings: Settings) -> AnswerGenerator:
    """Build a deterministic answer generator for repeatable evaluations."""

    chain = (
        answer_prompt()
        | ChatGoogleGenerativeAI(
            model=settings.llm_model,
            api_key=settings.google_api_key,
            temperature=0.0,
            max_retries=2,
        )
        | StrOutputParser()
    )

    def generate(question: str, documents: Sequence[Document]) -> str:
        return chain.invoke(
            {"question": question, "context": documents_to_context(documents)}
        )

    return generate


__all__ = [
    "RagJudgeGrade",
    "RagResult",
    "answer_prompt",
    "build_evaluation_answer_generator",
    "build_rag_chain",
    "build_rag_judge",
    "documents_to_context",
]

"""Retrieval configuration, corpus loading, and rank fusion."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sqlalchemy import create_engine, text

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


class RetrievalStrategy(StrEnum):
    """Retrieval configurations supported by the pipeline."""

    DENSE_ONLY = "dense-only"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid+rerank"


@dataclass(frozen=True)
class RetrievalConfig:
    """Tunable retrieval settings shared by chat and evaluation."""

    strategy: RetrievalStrategy
    final_k: int = 5
    candidate_k: int = 20
    dense_weight: float = 0.5
    bm25_weight: float = 0.5
    rrf_constant: int = 60

    def __post_init__(self) -> None:
        if self.final_k < 1 or self.candidate_k < self.final_k:
            raise ValueError("candidate_k must be greater than or equal to final_k")
        if self.dense_weight < 0 or self.bm25_weight < 0:
            raise ValueError("retrieval weights cannot be negative")
        if self.dense_weight + self.bm25_weight <= 0:
            raise ValueError("at least one retrieval weight must be positive")
        if self.rrf_constant < 0:
            raise ValueError("rrf_constant cannot be negative")


class DenseVectorStore(Protocol):
    """Small vector-store surface needed by the configurable retriever."""

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 4
    ) -> list[tuple[Document, float]]: ...


def tokenize(text_value: str) -> list[str]:
    """Apply the same deterministic tokenization to corpus and query."""

    return TOKEN_PATTERN.findall(text_value.lower())


def document_chunk_id(document: Document) -> str:
    """Return the stable chunk ID used for fusion and evaluation."""

    metadata_id = document.metadata.get("chunk_id")
    if metadata_id is not None and str(metadata_id):
        return str(metadata_id)
    if document.id is not None and str(document.id):
        return str(document.id)
    source = document.metadata.get("source", "unknown")
    chunk_index = document.metadata.get("chunk_index")
    if chunk_index is not None:
        return f"{source}:{chunk_index}"
    raise ValueError("retrieved document has no chunk_id, document ID, or chunk_index")


def load_collection_documents(
    connection_string: str, collection_name: str
) -> list[Document]:
    """Load the existing PGVector chunks for an in-memory BM25 index."""

    statement = text(
        """
        SELECT embedding.id, embedding.document, embedding.cmetadata
        FROM langchain_pg_embedding AS embedding
        JOIN langchain_pg_collection AS collection
          ON collection.uuid = embedding.collection_id
        WHERE collection.name = :collection_name
        ORDER BY embedding.id
        """
    )
    engine = create_engine(connection_string)
    try:
        with engine.connect() as connection:
            rows = connection.execute(statement, {"collection_name": collection_name})
            return [
                Document(
                    id=str(row.id),
                    page_content=row.document or "",
                    metadata=row.cmetadata or {},
                )
                for row in rows
            ]
    finally:
        engine.dispose()


def build_bm25_retriever(
    documents: Sequence[Document], *, k: int = 20
) -> BM25Retriever:
    """Build LangChain's BM25 retriever over stored chunks."""

    if not documents:
        raise ValueError("cannot build BM25 retriever: PGVector collection is empty")
    return BM25Retriever.from_documents(documents, preprocess_func=tokenize, k=k)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Document]],
    weights: Sequence[float],
    *,
    limit: int,
    rrf_constant: int = 60,
) -> list[Document]:
    """Fuse ranked lists by stable chunk ID using weighted RRF."""

    if len(ranked_lists) != len(weights):
        raise ValueError("one weight is required for each ranked list")
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    first_seen: dict[str, int] = {}
    for ranked, weight in zip(ranked_lists, weights, strict=True):
        for rank, document in enumerate(ranked, start=1):
            chunk_id = document_chunk_id(document)
            if chunk_id not in documents:
                documents[chunk_id] = document
                first_seen[chunk_id] = len(first_seen)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (
                rrf_constant + rank
            )
    ranked_ids = sorted(scores, key=lambda key: (-scores[key], first_seen[key]))
    return [documents[chunk_id] for chunk_id in ranked_ids[:limit]]

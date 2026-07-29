"""Retrieval pipeline orchestration and optional reranking."""

from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda

from note_rag.retrieval.core import (
    DenseVectorStore,
    RetrievalConfig,
    RetrievalStrategy,
    document_chunk_id,
    reciprocal_rank_fusion,
)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def print_retrieval_stage(
    stage: str,
    documents: Sequence[Document],
    *,
    scores: Sequence[float] | None = None,
    preview_length: int = 220,
) -> None:
    """Print a compact ranked list without PDF whitespace noise."""

    if scores is not None and len(scores) != len(documents):
        raise ValueError("one score is required for each printed document")
    print(f"\n{stage} ({len(documents)}):")
    for rank, document in enumerate(documents, start=1):
        metadata = document.metadata
        location = ", ".join(
            f"{key}={metadata[key]}"
            for key in ("source", "page", "chunk_index")
            if key in metadata
        )
        preview = " ".join(document.page_content.split())
        if len(preview) > preview_length:
            preview = f"{preview[:preview_length].rstrip()}..."
        score_text = "" if scores is None else f" similarity={scores[rank - 1]:.4f}"
        print(f"  {rank:>2}. id={document_chunk_id(document)}{score_text} ({location})")
        print(f"      {preview}")


class CrossEncoderReranker:
    """Lazily load and apply a sentence-transformers reranker."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, documents: Sequence[Document], k: int
    ) -> list[Document]:
        """Score query/chunk pairs and keep the highest-scoring chunks."""

        if not documents:
            return []
        pairs = [(query, document.page_content) for document in documents]
        scores = self._get_model().predict(pairs, show_progress_bar=False)
        ranked = sorted(
            enumerate(documents),
            key=lambda item: (-float(scores[item[0]]), item[0]),
        )
        return [document for _, document in ranked[:k]]


class RetrievalPipeline:
    """Run dense-only, hybrid, or hybrid plus cross-encoder retrieval."""

    def __init__(
        self,
        vector_store: DenseVectorStore,
        config: RetrievalConfig,
        *,
        bm25_retriever: Runnable[str, list[Document]] | None = None,
        reranker: CrossEncoderReranker | None = None,
        verbose: bool = False,
    ) -> None:
        if config.strategy is not RetrievalStrategy.DENSE_ONLY and not bm25_retriever:
            raise ValueError("hybrid retrieval requires a BM25 retriever")
        self.vector_store = vector_store
        self.config = config
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.verbose = verbose

    def _retrieve_dense(self, query: str, k: int) -> tuple[list[Document], list[float]]:
        results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
        documents = [document for document, _ in results]
        for document in documents:
            document.metadata.setdefault("chunk_id", document_chunk_id(document))
        return documents, [score for _, score in results]

    def _dense_candidates(self, query: str) -> list[Document]:
        documents, scores = self._retrieve_dense(query, self.config.candidate_k)
        if self.verbose:
            print_retrieval_stage("Dense vector candidates", documents, scores=scores)
        return documents

    def _bm25_candidates(self, query: str) -> list[Document]:
        assert self.bm25_retriever is not None
        documents = self.bm25_retriever.invoke(query)[: self.config.candidate_k]
        if self.verbose:
            print_retrieval_stage("BM25 candidates", documents)
        return documents

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve final context documents for one query."""

        if self.config.strategy is RetrievalStrategy.DENSE_ONLY:
            dense, scores = self._retrieve_dense(query, self.config.final_k)
            if self.verbose:
                print_retrieval_stage("Dense vector candidates", dense, scores=scores)
            return dense
        hybrid = reciprocal_rank_fusion(
            [self._dense_candidates(query), self._bm25_candidates(query)],
            [self.config.dense_weight, self.config.bm25_weight],
            limit=self.config.candidate_k,
            rrf_constant=self.config.rrf_constant,
        )
        if self.config.strategy is RetrievalStrategy.HYBRID:
            return hybrid[: self.config.final_k]
        self.reranker = self.reranker or CrossEncoderReranker()
        reranked = self.reranker.rerank(query, hybrid, self.config.final_k)
        if self.verbose:
            print_retrieval_stage("Cross-encoder reranked top 5", reranked)
        return reranked

    def as_runnable(self) -> Runnable[str, list[Document]]:
        """Expose retrieval as an LCEL runnable."""

        return RunnableLambda(self.retrieve)

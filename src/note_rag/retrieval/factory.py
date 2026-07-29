"""Factories for embeddings, vector storage, and retrieval pipelines."""

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

from note_rag.config import Settings
from note_rag.retrieval import (
    CrossEncoderReranker,
    RetrievalConfig,
    RetrievalPipeline,
    RetrievalStrategy,
    build_bm25_retriever,
    load_collection_documents,
)


def create_embeddings(settings: Settings) -> GoogleGenerativeAIEmbeddings:
    """Create Google's Gemini embedding client."""

    settings.require_api_key()
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.google_api_key,
        output_dimensionality=768,
    )


def create_vector_store(
    embeddings: GoogleGenerativeAIEmbeddings, settings: Settings
) -> PGVector:
    """Connect to the configured pgvector collection."""

    return PGVector(
        embeddings=embeddings,
        collection_name=settings.pgvector_collection,
        connection=settings.postgres_connection_string,
        use_jsonb=True,
        create_extension=True,
    )


def build_retrieval_pipelines(
    vector_store: PGVector, settings: Settings
) -> dict[str, RetrievalPipeline]:
    """Build all retrieval strategies with a shared corpus and reranker."""

    corpus = load_collection_documents(
        settings.postgres_connection_string,
        settings.pgvector_collection,
    )
    bm25_retriever = build_bm25_retriever(corpus)
    reranker = CrossEncoderReranker()
    return {
        strategy.value: RetrievalPipeline(
            vector_store,
            RetrievalConfig(strategy=strategy),
            bm25_retriever=(
                None if strategy is RetrievalStrategy.DENSE_ONLY else bm25_retriever
            ),
            reranker=(
                reranker if strategy is RetrievalStrategy.HYBRID_RERANK else None
            ),
        )
        for strategy in RetrievalStrategy
    }


def build_retrieval_pipeline(
    vector_store: PGVector,
    strategy: RetrievalStrategy,
    settings: Settings,
) -> RetrievalPipeline:
    """Build one retrieval strategy for interactive chat."""

    if strategy is RetrievalStrategy.DENSE_ONLY:
        return RetrievalPipeline(
            vector_store,
            RetrievalConfig(strategy=strategy),
            verbose=True,
        )
    corpus = load_collection_documents(
        settings.postgres_connection_string,
        settings.pgvector_collection,
    )
    return RetrievalPipeline(
        vector_store,
        RetrievalConfig(strategy=strategy),
        bm25_retriever=build_bm25_retriever(corpus),
        reranker=(
            CrossEncoderReranker()
            if strategy is RetrievalStrategy.HYBRID_RERANK
            else None
        ),
        verbose=True,
    )

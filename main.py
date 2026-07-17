"""Naive PDF RAG using Google Gemini and PostgreSQL with pgvector."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Any, TypedDict, cast

from dotenv import dotenv_values
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

PROJECT_ROOT = Path(__file__).resolve().parent
ENV = dotenv_values(PROJECT_ROOT / ".env")


def env_value(name: str, default: str = "") -> str:
    """Read a string from the project-root .env file."""

    return ENV.get(name) or default


GOOGLE_API_KEY = SecretStr(env_value("GOOGLE_API_KEY"))
POSTGRES_CONNECTION_STRING = env_value(
    "POSTGRES_CONNECTION_STRING",
    "postgresql+psycopg://langchain:langchain@localhost:6024/langchain",
)
PGVECTOR_COLLECTION = env_value("PGVECTOR_COLLECTION", "phase1_pdf_rag")
EMBEDDING_MODEL = env_value("EMBEDDING_MODEL", "gemini-embedding-2")
LLM_MODEL = env_value("LLM_MODEL", "gemini-3.5-flash")


class RagResult(TypedDict):
    """Values returned after retrieval and answer generation."""

    question: str
    source_documents: list[Document]
    answer: str


class CliArgs(argparse.Namespace):
    """Typed command-line arguments."""

    pdf: Path | None


def pdf_path(value: str) -> Path:
    """Validate the positional PDF argument for argparse."""

    path = Path(value)
    if path.suffix.lower() != ".pdf":
        raise argparse.ArgumentTypeError("input must be a .pdf file")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"PDF not found: {path}")
    return path.resolve()


def parse_args() -> CliArgs:
    """Parse an optional PDF to ingest before chat starts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pdf",
        nargs="?",
        type=pdf_path,
        help="Optional local PDF file to ingest before chat",
    )
    return parser.parse_args(namespace=CliArgs())


def create_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create Google's Gemini embedding client."""

    if not GOOGLE_API_KEY.get_secret_value():
        raise RuntimeError("GOOGLE_API_KEY is missing from .env")
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=GOOGLE_API_KEY,
        output_dimensionality=768,
    )


def create_vector_store(
    embeddings: GoogleGenerativeAIEmbeddings,
) -> PGVector:
    """Connect to the pgvector collection in local PostgreSQL."""

    return PGVector(
        embeddings=embeddings,
        collection_name=PGVECTOR_COLLECTION,
        connection=POSTGRES_CONNECTION_STRING,
        use_jsonb=True,
        create_extension=True,
    )


def load_and_chunk_pdf(pdf: Path) -> list[Document]:
    """Load PDF pages and split them into 700-character overlapping chunks."""

    # Loading stage: PyPDFLoader creates one LangChain Document per PDF page.
    # Import lazily because the community loader is needed only during ingestion.
    from langchain_community.document_loaders import PyPDFLoader

    pages = PyPDFLoader(str(pdf)).load()

    # Chunking stage: split pages while retaining their page metadata.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = splitter.split_documents(pages)

    for chunk_index, chunk in enumerate(chunks):
        chunk.metadata = {
            **chunk.metadata,
            "source": pdf.name,
            "module": pdf.stem,
            "chunk_index": chunk_index,
        }
    return chunks


def chunk_id(pdf: Path, chunk: Document) -> str:
    """Create a stable ID so repeated ingestion upserts the same chunk."""

    identity = f"{pdf.as_posix()}:{chunk.metadata['chunk_index']}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


def ingest_pdf(vector_store: PGVector, pdf: Path) -> int:
    """Embed and upsert all chunks from one PDF into PGVector."""

    chunks = load_and_chunk_pdf(pdf)
    if not chunks:
        raise ValueError(f"No extractable text found in {pdf.name}")

    # Storage stage: PGVector embeds chunks and stores vectors plus metadata.
    vector_store.add_documents(
        documents=chunks,
        ids=[chunk_id(pdf, chunk) for chunk in chunks],
    )
    return len(chunks)


def print_retrieved_chunks(documents: list[Document]) -> list[Document]:
    """Print top-5 retrieval results and pass them unchanged to generation."""

    print(f"\nRetrieved chunks ({len(documents)}):")
    for position, document in enumerate(documents, start=1):
        print(f"\n--- Chunk {position} ---")
        print(f"Metadata: {document.metadata}")
        print(document.page_content.strip())
    return documents


def format_prompt_values(inputs: dict[str, Any]) -> dict[str, str]:
    """Convert retrieved documents into prompt variables."""

    documents = cast("list[Document]", inputs["source_documents"])
    context = "\n\n--- Retrieved chunk ---\n\n".join(
        document.page_content for document in documents
    )
    return {"question": cast("str", inputs["question"]), "context": context}


def build_rag_chain(vector_store: PGVector) -> Runnable[str, RagResult]:
    """Build top-5 similarity retrieval and Gemini generation with LCEL."""

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    retrieval = retriever | RunnableLambda(print_retrieved_chunks)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's question using the retrieved PDF context."
                "\n\nContext:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        api_key=GOOGLE_API_KEY,
        temperature=1.0,
        max_retries=2,
    )
    generation = (
        RunnableLambda(format_prompt_values) | prompt | llm | StrOutputParser()
    )

    chain = (
        RunnableParallel(
            question=RunnablePassthrough(),
            source_documents=retrieval,
        )
        | RunnablePassthrough.assign(answer=generation)
    )
    return cast("Runnable[str, RagResult]", chain)


def chat(chain: Runnable[str, RagResult]) -> None:
    """Run the interactive PDF question loop."""

    print("\nPDF RAG is ready. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return
        if not question:
            continue

        result = chain.invoke(question)
        print(f"\nAnswer:\n{result['answer']}")


def main() -> int:
    """Optionally ingest a PDF, then start retrieval-augmented chat."""

    args = parse_args()
    try:
        vector_store = create_vector_store(create_embeddings())
        if args.pdf is not None:
            chunk_count = ingest_pdf(vector_store, args.pdf)
            print(
                f"Upserted {chunk_count} chunks from {args.pdf.name} into PGVector."
            )
        else:
            print("No PDF supplied; using the existing PGVector collection.")
        chat(build_rag_chain(vector_store))
        return 0
    except (RuntimeError, ValueError) as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

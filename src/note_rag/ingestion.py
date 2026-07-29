"""PDF loading, chunking, and vector-store ingestion."""

from __future__ import annotations

import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk_pdf(pdf: Path) -> list[Document]:
    """Load PDF pages and split them into overlapping text chunks."""

    from langchain_community.document_loaders import PyPDFLoader

    pages = PyPDFLoader(str(pdf)).load()
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
    vector_store.add_documents(
        documents=chunks,
        ids=[chunk_id(pdf, chunk) for chunk in chunks],
    )
    return len(chunks)


def ingest_pdfs(vector_store: PGVector, pdfs: list[Path]) -> list[tuple[Path, int]]:
    """Ingest multiple PDFs sequentially and return each chunk count."""

    return [(pdf, ingest_pdf(vector_store, pdf)) for pdf in pdfs]

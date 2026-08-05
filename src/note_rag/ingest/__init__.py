"""Synchronous document ingestion for Phase 3."""

from note_rag.ingest.parsers import (
    MarkdownParser,
    ParsedDocument,
    ParserRegistry,
    PdfParser,
    TextParser,
)
from note_rag.ingest.pipeline import IngestionPipeline, IngestionResult
from note_rag.ingest.storage import LocalFileStorage, StoredFile

__all__ = [
    "IngestionPipeline",
    "IngestionResult",
    "LocalFileStorage",
    "MarkdownParser",
    "ParsedDocument",
    "ParserRegistry",
    "PdfParser",
    "StoredFile",
    "TextParser",
]

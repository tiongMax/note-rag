"""Small parsers for the Phase 3 document types."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

from pypdf import PdfReader

from note_rag.ingest.errors import (
    DocumentParsingError,
    UnsupportedDocumentTypeError,
)


@dataclass(frozen=True, slots=True)
class SourceSpan:
    char_start: int
    char_end: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_spans: tuple[SourceSpan, ...] = ()

    def metadata_for_range(
        self,
        char_start: int,
        char_end: int,
    ) -> dict[str, Any]:
        result = dict(self.metadata)
        matching_pages = [
            span.metadata["page_number"]
            for span in self.source_spans
            if "page_number" in span.metadata
            and span.char_start < char_end
            and span.char_end > char_start
        ]
        if len(matching_pages) == 1:
            result["page_number"] = matching_pages[0]
        elif matching_pages:
            result["page_numbers"] = matching_pages
        return result


class DocumentParser(Protocol):
    def parse(self, content: bytes) -> ParsedDocument: ...


def _decode_utf8(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError as error:
        raise DocumentParsingError("document is not valid UTF-8 text") from error


class TextParser:
    def parse(self, content: bytes) -> ParsedDocument:
        text = _decode_utf8(content)
        if not text.strip():
            raise DocumentParsingError("document contains no text")
        return ParsedDocument(
            text=text,
            metadata={"format": "text"},
            source_spans=(SourceSpan(0, len(text)),),
        )


class MarkdownParser:
    def parse(self, content: bytes) -> ParsedDocument:
        text = _decode_utf8(content)
        if not text.strip():
            raise DocumentParsingError("document contains no Markdown text")
        return ParsedDocument(
            text=text,
            metadata={"format": "markdown"},
            source_spans=(SourceSpan(0, len(text)),),
        )


class PdfParser:
    def parse(self, content: bytes) -> ParsedDocument:
        try:
            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted and reader.decrypt("") == 0:
                raise DocumentParsingError("encrypted PDF cannot be read")

            parts: list[str] = []
            spans: list[SourceSpan] = []
            current_length = 0
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = (page.extract_text() or "").strip()
                if not page_text:
                    continue
                separator = "\n\n" if parts else ""
                parts.append(separator + page_text)
                char_start = current_length + len(separator)
                char_end = char_start + len(page_text)
                spans.append(
                    SourceSpan(
                        char_start,
                        char_end,
                        {"page_number": page_number},
                    )
                )
                current_length = char_end
        except DocumentParsingError:
            raise
        except Exception as error:
            raise DocumentParsingError("PDF could not be parsed") from error

        text = "".join(parts)
        if not text:
            raise DocumentParsingError("PDF contains no extractable text")
        return ParsedDocument(
            text=text,
            metadata={"format": "pdf", "page_count": len(reader.pages)},
            source_spans=tuple(spans),
        )


class ParserRegistry:
    """Resolve the supported parser from a sanitized filename suffix."""

    def __init__(self) -> None:
        self._parsers: dict[str, DocumentParser] = {
            ".txt": TextParser(),
            ".md": MarkdownParser(),
            ".markdown": MarkdownParser(),
            ".pdf": PdfParser(),
        }

    @property
    def supported_suffixes(self) -> frozenset[str]:
        return frozenset(self._parsers)

    def get(self, filename: str) -> DocumentParser:
        suffix = Path(filename).suffix.lower()
        parser = self._parsers.get(suffix)
        if parser is None:
            supported = ", ".join(sorted(self._parsers))
            raise UnsupportedDocumentTypeError(
                f"unsupported document type; expected one of: {supported}"
            )
        return parser

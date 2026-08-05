from types import SimpleNamespace

import pytest

from note_rag.ingest.errors import (
    DocumentParsingError,
    UnsupportedDocumentTypeError,
)
from note_rag.ingest.parsers import (
    MarkdownParser,
    ParserRegistry,
    PdfParser,
    TextParser,
)


def test_text_parser_normalizes_newlines() -> None:
    parsed = TextParser().parse(b"\xef\xbb\xbfalpha\r\nbeta\rgamma")

    assert parsed.text == "alpha\nbeta\ngamma"
    assert parsed.metadata == {"format": "text"}


def test_markdown_parser_preserves_markup() -> None:
    parsed = MarkdownParser().parse(b"# Heading\n\n- item")

    assert parsed.text == "# Heading\n\n- item"
    assert parsed.metadata == {"format": "markdown"}


def test_text_parser_rejects_non_utf8_content() -> None:
    with pytest.raises(DocumentParsingError, match="valid UTF-8"):
        TextParser().parse(b"\xff\xfe\xfa")


def test_pdf_parser_preserves_page_spans(monkeypatch) -> None:
    pages = [
        SimpleNamespace(extract_text=lambda: "first page"),
        SimpleNamespace(extract_text=lambda: "second page"),
    ]
    fake_reader = SimpleNamespace(
        pages=pages,
        is_encrypted=False,
    )
    monkeypatch.setattr(
        "note_rag.ingest.parsers.PdfReader",
        lambda _stream: fake_reader,
    )

    parsed = PdfParser().parse(b"fake-pdf")

    assert parsed.text == "first page\n\nsecond page"
    assert parsed.metadata == {"format": "pdf", "page_count": 2}
    assert parsed.metadata_for_range(0, 10)["page_number"] == 1
    assert parsed.metadata_for_range(8, 20)["page_numbers"] == [1, 2]


def test_parser_registry_rejects_unknown_suffix() -> None:
    with pytest.raises(UnsupportedDocumentTypeError, match="unsupported"):
        ParserRegistry().get("notes.docx")

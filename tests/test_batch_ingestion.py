"""Unit tests for sequential batch PDF ingestion."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from note_rag import ingestion


class BatchIngestionTests(unittest.TestCase):
    def test_ingest_pdfs_processes_each_file_in_order(self) -> None:
        pdfs = [Path("first.pdf"), Path("second.pdf"), Path("third.pdf")]
        vector_store = object()

        with patch("note_rag.ingestion.ingest_pdf", side_effect=[2, 3, 4]) as ingest:
            results = ingestion.ingest_pdfs(vector_store, pdfs)  # type: ignore[arg-type]

        self.assertEqual(results, list(zip(pdfs, [2, 3, 4], strict=True)))
        self.assertEqual(
            [call.args for call in ingest.call_args_list],
            [(vector_store, pdf) for pdf in pdfs],
        )

    def test_ingest_pdfs_accepts_an_empty_batch(self) -> None:
        self.assertEqual(
            ingestion.ingest_pdfs(object(), []),  # type: ignore[arg-type]
            [],
        )


if __name__ == "__main__":
    unittest.main()

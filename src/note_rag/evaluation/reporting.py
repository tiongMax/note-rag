"""Public evaluation reporting API."""

from note_rag.evaluation.csv import (
    save_comparison_csv,
    save_generation_csvs,
    save_retrieval_details_csv,
)
from note_rag.evaluation.tables import print_comparison_table, print_generation_table

__all__ = [
    "print_comparison_table",
    "print_generation_table",
    "save_comparison_csv",
    "save_generation_csvs",
    "save_retrieval_details_csv",
]

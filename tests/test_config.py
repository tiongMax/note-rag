import os
from pathlib import Path

from note_rag.config import load_environment


def test_loads_values_from_explicit_dotenv(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("NOTE_RAG_DOTENV_TEST", raising=False)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "NOTE_RAG_DOTENV_TEST=from-file\n",
        encoding="utf-8",
    )

    assert load_environment(dotenv_path) is True
    assert os.environ["NOTE_RAG_DOTENV_TEST"] == "from-file"


def test_process_environment_takes_precedence(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("NOTE_RAG_DOTENV_TEST", "from-process")
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "NOTE_RAG_DOTENV_TEST=from-file\n",
        encoding="utf-8",
    )

    load_environment(dotenv_path)

    assert os.environ["NOTE_RAG_DOTENV_TEST"] == "from-process"

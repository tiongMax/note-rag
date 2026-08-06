import os
from pathlib import Path

import pytest

from note_rag.api.settings import ApiSettings
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


def test_rejects_insecure_production_configuration() -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        ApiSettings(app_environment="production")


def test_rejects_unknown_chunking_strategy() -> None:
    with pytest.raises(ValueError, match="CHUNKING_STRATEGY"):
        ApiSettings(chunking_strategy="semantic")


def test_reads_cross_encoder_configuration(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_BACKEND", "cross_encoder")
    monkeypatch.setenv("CROSS_ENCODER_MODEL", "local-reranker")
    monkeypatch.setenv("CROSS_ENCODER_DEVICE", "cpu")
    monkeypatch.setenv("CROSS_ENCODER_BATCH_SIZE", "8")

    settings = ApiSettings.from_env()

    assert settings.reranker_backend == "cross_encoder"
    assert settings.cross_encoder_model == "local-reranker"
    assert settings.cross_encoder_device == "cpu"
    assert settings.cross_encoder_batch_size == 8


def test_rejects_unknown_reranker_backend() -> None:
    with pytest.raises(ValueError, match="RERANKER_BACKEND"):
        ApiSettings(reranker_backend="hosted")


def test_accepts_secure_production_configuration() -> None:
    settings = ApiSettings(
        app_environment="production",
        gemini_api_key="configured",
        api_auth_token="a-production-token-with-adequate-length",
        allowed_hosts=("rag.example.com",),
    )

    assert settings.app_environment == "production"

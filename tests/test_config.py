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


def test_reads_and_validates_bm25_configuration(monkeypatch) -> None:
    monkeypatch.setenv("RETRIEVAL_LEXICAL_BACKEND", "postgres_fts")
    monkeypatch.setenv("RETRIEVAL_BM25_K1", "1.8")
    monkeypatch.setenv("RETRIEVAL_BM25_B", "0.6")

    settings = ApiSettings.from_env()

    assert settings.retrieval_lexical_backend == "postgres_fts"
    assert settings.retrieval_bm25_k1 == 1.8
    assert settings.retrieval_bm25_b == 0.6
    with pytest.raises(ValueError, match="RETRIEVAL_LEXICAL_BACKEND"):
        ApiSettings(retrieval_lexical_backend="elasticsearch")


def test_accepts_secure_production_configuration() -> None:
    settings = ApiSettings(
        app_environment="production",
        gemini_api_key="configured",
        api_auth_token="a-production-token-with-adequate-length",
        allowed_hosts=("rag.example.com",),
    )

    assert settings.app_environment == "production"


def test_boolean_environment_values_are_strict(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENVIRONMENT", "development")
    monkeypatch.setenv("GUARDRAILS_ENABLED", "ture")

    with pytest.raises(ValueError, match="GUARDRAILS_ENABLED must be a boolean"):
        ApiSettings.from_env()


def test_boolean_environment_values_accept_explicit_false(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENVIRONMENT", "development")
    monkeypatch.setenv("GUARDRAILS_ENABLED", "off")

    assert ApiSettings.from_env().guardrails_enabled is False


def test_normalizes_and_validates_application_environment() -> None:
    settings = ApiSettings(
        app_environment="  ProD  ",
        gemini_api_key="configured",
        api_auth_token="a-production-token-with-adequate-length",
        allowed_hosts=("rag.example.com",),
    )

    assert settings.app_environment == "production"
    with pytest.raises(ValueError, match="APP_ENVIRONMENT"):
        ApiSettings(app_environment="staging")


def test_production_requires_guardrails_without_explicit_break_glass() -> None:
    production = {
        "app_environment": "production",
        "gemini_api_key": "configured",
        "api_auth_token": "a-production-token-with-adequate-length",
        "allowed_hosts": ("rag.example.com",),
        "guardrails_enabled": False,
    }

    with pytest.raises(ValueError, match="ALLOW_UNGUARDED_CHAT_IN_PRODUCTION"):
        ApiSettings(**production)

    settings = ApiSettings(
        **production,
        allow_unguarded_chat_in_production=True,
    )
    assert settings.guardrails_enabled is False


@pytest.mark.parametrize(
    ("provider_max", "hard_max", "expected"),
    [(2048, 512, 512), (256, 512, 256)],
)
def test_effective_chat_output_limit_uses_stricter_cap(
    provider_max: int,
    hard_max: int,
    expected: int,
) -> None:
    settings = ApiSettings(
        chat_max_output_tokens=provider_max,
        chat_output_hard_max_tokens=hard_max,
    )

    assert settings.effective_chat_max_output_tokens == expected


def test_reads_and_validates_conversation_memory_configuration(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CHAT_MEMORY_RECENT_TURNS", "5")
    monkeypatch.setenv("CHAT_MEMORY_SEMANTIC_K", "2")
    monkeypatch.setenv("CHAT_MEMORY_SEMANTIC_MIN_SIMILARITY", "0.35")
    monkeypatch.setenv("CHAT_MEMORY_SUMMARY_MAX_TOKENS", "72")

    settings = ApiSettings.from_env()

    assert settings.chat_memory_recent_turns == 5
    assert settings.chat_memory_semantic_k == 2
    assert settings.chat_memory_semantic_min_similarity == 0.35
    assert settings.chat_memory_summary_max_tokens == 72
    with pytest.raises(ValueError, match="CHAT_MEMORY_SEMANTIC_K"):
        ApiSettings(chat_memory_semantic_k=-1)
    with pytest.raises(ValueError, match="CHAT_MEMORY_SEMANTIC_MIN_SIMILARITY"):
        ApiSettings(chat_memory_semantic_min_similarity=2.0)

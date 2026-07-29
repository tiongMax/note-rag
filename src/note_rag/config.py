"""Application configuration loaded from the project environment."""

from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values
from pydantic import SecretStr

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Runtime settings for models and PostgreSQL."""

    google_api_key: SecretStr
    postgres_connection_string: str
    pgvector_collection: str
    embedding_model: str
    llm_model: str
    eval_judge_model: str

    @classmethod
    def from_env(cls, path: Path = PROJECT_ROOT / ".env") -> "Settings":
        values = dotenv_values(path)

        def value(name: str, default: str = "") -> str:
            return values.get(name) or default

        llm_model = value("LLM_MODEL", "gemini-3.5-flash")
        return cls(
            google_api_key=SecretStr(value("GOOGLE_API_KEY")),
            postgres_connection_string=value(
                "POSTGRES_CONNECTION_STRING",
                "postgresql+psycopg://langchain:langchain@localhost:6024/langchain",
            ),
            pgvector_collection=value("PGVECTOR_COLLECTION", "phase1_pdf_rag"),
            embedding_model=value("EMBEDDING_MODEL", "gemini-embedding-2"),
            llm_model=llm_model,
            eval_judge_model=value("EVAL_JUDGE_MODEL", llm_model),
        )

    def require_api_key(self) -> None:
        """Fail early when a Gemini-backed operation has no API key."""

        if not self.google_api_key.get_secret_value():
            raise RuntimeError("GOOGLE_API_KEY is missing from .env")


settings = Settings.from_env()

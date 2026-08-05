"""Persistence configuration."""

import os
from dataclasses import dataclass

DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
)


@dataclass(frozen=True, slots=True)
class DatabaseSettings:
    url: str = DEFAULT_DATABASE_URL
    echo: bool = False

    @classmethod
    def from_env(cls) -> "DatabaseSettings":
        url = os.getenv("DATABASE_URL") or os.getenv(
            "POSTGRES_CONNECTION_STRING",
            DEFAULT_DATABASE_URL,
        )
        echo = os.getenv("DATABASE_ECHO", "").lower() in {"1", "true", "yes"}
        return cls(url=url, echo=echo)

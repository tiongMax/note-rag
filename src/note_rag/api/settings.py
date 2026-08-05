"""Environment-backed settings for the Phase 1 API."""

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ApiSettings:
    app_name: str = "Note RAG"
    app_environment: str = "development"
    chunk_size: int = 200
    chunk_overlap: int = 20

    @classmethod
    def from_env(cls) -> "ApiSettings":
        return cls(
            app_name=os.getenv("APP_NAME", "Note RAG"),
            app_environment=os.getenv("APP_ENVIRONMENT", "development"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "200")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "20")),
        )


api_settings = ApiSettings.from_env()

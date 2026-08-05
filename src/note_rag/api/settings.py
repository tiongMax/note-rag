"""Environment-backed settings for the Phase 1 API."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ApiSettings:
    app_name: str = "Note RAG"
    app_environment: str = "development"
    chunk_size: int = 200
    chunk_overlap: int = 20
    storage_path: Path = Path("data/uploads")
    max_upload_bytes: int = 10 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "ApiSettings":
        return cls(
            app_name=os.getenv("APP_NAME", "Note RAG"),
            app_environment=os.getenv("APP_ENVIRONMENT", "development"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "200")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "20")),
            storage_path=Path(os.getenv("UPLOAD_STORAGE_PATH", "data/uploads")),
            max_upload_bytes=int(
                os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))
            ),
        )


api_settings = ApiSettings.from_env()

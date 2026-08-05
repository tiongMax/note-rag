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
    embedding_model: str = "gemini-embedding-2"
    embedding_dimension: int = 768
    embedding_batch_size: int = 32
    retrieval_candidate_multiplier: int = 4
    retrieval_rrf_k: int = 60
    gemini_api_key: str = ""

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
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "gemini-embedding-2",
            ),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "768")),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
            retrieval_candidate_multiplier=int(
                os.getenv("RETRIEVAL_CANDIDATE_MULTIPLIER", "4")
            ),
            retrieval_rrf_k=int(os.getenv("RETRIEVAL_RRF_K", "60")),
            gemini_api_key=(
                os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_API_KEY")
                or ""
            ),
        )


api_settings = ApiSettings.from_env()

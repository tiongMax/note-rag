"""Environment-backed application settings."""

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True, slots=True)
class ApiSettings:
    app_name: str = "Note RAG"
    app_environment: str = "development"
    chunk_size: int = 200
    chunk_overlap: int = 20
    storage_path: Path = Path("data/uploads")
    frontend_dist_path: Path = Path("frontend/dist")
    max_upload_bytes: int = 10 * 1024 * 1024
    embedding_model: str = "gemini-embedding-2"
    embedding_dimension: int = 768
    embedding_batch_size: int = 32
    retrieval_candidate_multiplier: int = 4
    retrieval_rrf_k: int = 60
    context_candidate_k: int = 20
    context_max_chunks: int = 8
    context_max_tokens: int = 1200
    rerank_weight: float = 0.7
    chat_model: str = "gemini-3.5-flash"
    chat_temperature: float = 0.1
    chat_max_output_tokens: int = 1024
    chat_history_max_messages: int = 20
    chat_history_max_tokens: int = 2000
    background_worker_enabled: bool = True
    worker_max_attempts: int = 3
    worker_retry_backoff_seconds: float = 2.0
    worker_poll_interval_seconds: float = 1.0
    worker_lease_timeout_seconds: float = 300.0
    gemini_api_key: str = ""
    api_auth_token: str = ""
    allowed_origins: tuple[str, ...] = (
        "http://127.0.0.1:5174",
        "http://localhost:5174",
    )
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "testserver")
    max_request_bytes: int = 12 * 1024 * 1024
    rate_limit_requests: int = 120
    rate_limit_window_seconds: int = 60
    log_level: str = "INFO"
    json_logs: bool = True
    metrics_enabled: bool = True

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be greater than zero")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                "CHUNK_OVERLAP must be non-negative and smaller than CHUNK_SIZE"
            )
        positive_values = {
            "MAX_UPLOAD_BYTES": self.max_upload_bytes,
            "EMBEDDING_DIMENSION": self.embedding_dimension,
            "EMBEDDING_BATCH_SIZE": self.embedding_batch_size,
            "CHAT_MAX_OUTPUT_TOKENS": self.chat_max_output_tokens,
            "MAX_REQUEST_BYTES": self.max_request_bytes,
            "RATE_LIMIT_REQUESTS": self.rate_limit_requests,
            "RATE_LIMIT_WINDOW_SECONDS": self.rate_limit_window_seconds,
            "WORKER_MAX_ATTEMPTS": self.worker_max_attempts,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.max_request_bytes < self.max_upload_bytes:
            raise ValueError(
                "MAX_REQUEST_BYTES must be at least MAX_UPLOAD_BYTES"
            )
        if self.log_level.upper() not in {
            "CRITICAL",
            "ERROR",
            "WARNING",
            "INFO",
            "DEBUG",
        }:
            raise ValueError("LOG_LEVEL is invalid")
        if self.app_environment.lower() in {"production", "prod"}:
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is required in production")
            if len(self.api_auth_token) < 24:
                raise ValueError(
                    "API_AUTH_TOKEN must contain at least 24 characters "
                    "in production"
                )
            if "*" in self.allowed_hosts:
                raise ValueError(
                    "ALLOWED_HOSTS cannot contain '*' in production"
                )

    @classmethod
    def from_env(cls) -> "ApiSettings":
        return cls(
            app_name=os.getenv("APP_NAME", "Note RAG"),
            app_environment=os.getenv("APP_ENVIRONMENT", "development"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "200")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "20")),
            storage_path=Path(os.getenv("UPLOAD_STORAGE_PATH", "data/uploads")),
            frontend_dist_path=Path(
                os.getenv("FRONTEND_DIST_PATH", "frontend/dist")
            ),
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
            context_candidate_k=int(os.getenv("CONTEXT_CANDIDATE_K", "20")),
            context_max_chunks=int(os.getenv("CONTEXT_MAX_CHUNKS", "8")),
            context_max_tokens=int(os.getenv("CONTEXT_MAX_TOKENS", "1200")),
            rerank_weight=float(os.getenv("RERANK_WEIGHT", "0.7")),
            chat_model=os.getenv("CHAT_MODEL", "gemini-3.5-flash"),
            chat_temperature=float(os.getenv("CHAT_TEMPERATURE", "0.1")),
            chat_max_output_tokens=int(
                os.getenv("CHAT_MAX_OUTPUT_TOKENS", "1024")
            ),
            chat_history_max_messages=int(
                os.getenv("CHAT_HISTORY_MAX_MESSAGES", "20")
            ),
            chat_history_max_tokens=int(
                os.getenv("CHAT_HISTORY_MAX_TOKENS", "2000")
            ),
            background_worker_enabled=_env_bool(
                "BACKGROUND_WORKER_ENABLED",
                True,
            ),
            worker_max_attempts=int(os.getenv("WORKER_MAX_ATTEMPTS", "3")),
            worker_retry_backoff_seconds=float(
                os.getenv("WORKER_RETRY_BACKOFF_SECONDS", "2")
            ),
            worker_poll_interval_seconds=float(
                os.getenv("WORKER_POLL_INTERVAL_SECONDS", "1")
            ),
            worker_lease_timeout_seconds=float(
                os.getenv("WORKER_LEASE_TIMEOUT_SECONDS", "300")
            ),
            gemini_api_key=(
                os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_API_KEY")
                or ""
            ),
            api_auth_token=os.getenv("API_AUTH_TOKEN", ""),
            allowed_origins=_env_csv(
                "ALLOWED_ORIGINS",
                (
                    "http://127.0.0.1:5174",
                    "http://localhost:5174",
                ),
            ),
            allowed_hosts=_env_csv(
                "ALLOWED_HOSTS",
                ("127.0.0.1", "localhost", "testserver"),
            ),
            max_request_bytes=int(
                os.getenv("MAX_REQUEST_BYTES", str(12 * 1024 * 1024))
            ),
            rate_limit_requests=int(
                os.getenv("RATE_LIMIT_REQUESTS", "120")
            ),
            rate_limit_window_seconds=int(
                os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")
            ),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            json_logs=_env_bool("JSON_LOGS", True),
            metrics_enabled=_env_bool("METRICS_ENABLED", True),
        )


api_settings = ApiSettings.from_env()

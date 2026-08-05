"""Security, request-limit, rate-limit, and access-log middleware."""

import logging
import re
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response

from note_rag.api.errors import error_response
from note_rag.api.observability import MetricsRegistry
from note_rag.api.settings import ApiSettings

logger = logging.getLogger("note_rag.access")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class RateLimiter:
    """Small in-memory limiter for a single-process deployment."""

    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._requests: defaultdict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, now: float | None = None) -> tuple[bool, int]:
        current = now if now is not None else time.monotonic()
        cutoff = current - self.window_seconds
        with self._lock:
            entries = self._requests[key]
            while entries and entries[0] <= cutoff:
                entries.popleft()
            if len(entries) >= self.limit:
                retry_after = max(1, int(entries[0] + self.window_seconds - current))
                return False, retry_after
            entries.append(current)
            return True, 0


def install_http_middleware(
    app: FastAPI,
    settings: ApiSettings,
    metrics: MetricsRegistry,
) -> None:
    limiter = RateLimiter(
        settings.rate_limit_requests,
        settings.rate_limit_window_seconds,
    )

    @app.middleware("http")
    async def production_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        started = time.perf_counter()
        supplied_request_id = request.headers.get("x-request-id", "")
        request_id = (
            supplied_request_id
            if _REQUEST_ID.fullmatch(supplied_request_id)
            else uuid.uuid4().hex
        )
        request.state.request_id = request_id
        metrics.start_request()
        response: Response | None = None
        route = request.url.path
        try:
            content_length = request.headers.get("content-length")
            if (
                content_length is not None
                and content_length.isdigit()
                and int(content_length) > settings.max_request_bytes
            ):
                response = error_response(
                    request,
                    status_code=413,
                    code="request_too_large",
                    message=(
                        f"Request exceeds the {settings.max_request_bytes}-byte "
                        "limit."
                    ),
                )
            else:
                client = request.client.host if request.client else "unknown"
                rate_limited = _rate_limited_path(request)
                allowed, retry_after = (
                    limiter.allow(client) if rate_limited else (True, 0)
                )
                if not allowed:
                    response = error_response(
                        request,
                        status_code=429,
                        code="rate_limit_exceeded",
                        message="Too many requests. Please retry later.",
                        headers={"Retry-After": str(retry_after)},
                    )
                elif _requires_authentication(request) and not _authenticated(
                    request,
                    settings.api_auth_token,
                ):
                    response = error_response(
                        request,
                        status_code=401,
                        code="unauthorized",
                        message="A valid API bearer token is required.",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                else:
                    response = await call_next(request)
                    matched_route = request.scope.get("route")
                    route = getattr(matched_route, "path", route)
            return response
        finally:
            status_code = response.status_code if response is not None else 500
            duration = time.perf_counter() - started
            metrics.finish_request(
                method=request.method,
                route=route,
                status_code=status_code,
                duration_seconds=duration,
            )
            logger.info(
                "request.complete",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "route": route,
                    "status_code": status_code,
                    "duration_ms": round(duration * 1000, 3),
                    "client": request.client.host if request.client else None,
                },
            )
            if response is not None:
                _set_security_headers(response, request_id)


def _requires_authentication(request: Request) -> bool:
    if request.method == "OPTIONS":
        return False
    return request.url.path.startswith("/api/v1")


def _authenticated(request: Request, expected_token: str) -> bool:
    if not expected_token:
        return True
    authorization = request.headers.get("authorization", "")
    supplied = ""
    if authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if not supplied:
        supplied = request.headers.get("x-api-key", "")
    return bool(supplied) and secrets.compare_digest(supplied, expected_token)


def _rate_limited_path(request: Request) -> bool:
    return request.url.path.startswith("/api/") or request.url.path == "/health/ready"


def _set_security_headers(response: Response, request_id: str) -> None:
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=()"
    )
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'"
    )

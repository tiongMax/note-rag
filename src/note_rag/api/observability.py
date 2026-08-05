"""Structured logging and lightweight Prometheus-compatible metrics."""

import json
import logging
import threading
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    """Render one structured JSON object per log record."""

    _standard_fields = frozenset(
        logging.LogRecord("", 0, "", 0, "", (), None).__dict__
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._standard_fields and key not in {
                "message",
                "asctime",
            }:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str, *, json_logs: bool) -> None:
    """Configure application logging once per process."""

    root = logging.getLogger()
    if getattr(root, "_note_rag_configured", False):
        root.setLevel(level)
        return
    handler = logging.StreamHandler()
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s"
            )
        )
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root._note_rag_configured = True  # type: ignore[attr-defined]


class MetricsRegistry:
    """Thread-safe request counters suitable for a single API process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at = time.monotonic()
        self._requests: Counter[tuple[str, str, int]] = Counter()
        self._duration_seconds: defaultdict[tuple[str, str], float] = defaultdict(
            float
        )
        self._in_progress = 0

    def start_request(self) -> None:
        with self._lock:
            self._in_progress += 1

    def finish_request(
        self,
        *,
        method: str,
        route: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        with self._lock:
            self._in_progress = max(0, self._in_progress - 1)
            self._requests[(method, route, status_code)] += 1
            self._duration_seconds[(method, route)] += duration_seconds

    def render(self) -> str:
        with self._lock:
            requests = self._requests.copy()
            durations = self._duration_seconds.copy()
            in_progress = self._in_progress
            uptime = time.monotonic() - self._started_at

        lines = [
            "# HELP note_rag_uptime_seconds Process uptime in seconds.",
            "# TYPE note_rag_uptime_seconds gauge",
            f"note_rag_uptime_seconds {uptime:.6f}",
            "# HELP note_rag_http_requests_in_progress Active HTTP requests.",
            "# TYPE note_rag_http_requests_in_progress gauge",
            f"note_rag_http_requests_in_progress {in_progress}",
            "# HELP note_rag_http_requests_total Total HTTP requests.",
            "# TYPE note_rag_http_requests_total counter",
        ]
        for (method, route, status_code), count in sorted(requests.items()):
            labels = _labels(
                method=method,
                route=route,
                status=str(status_code),
            )
            lines.append(f"note_rag_http_requests_total{{{labels}}} {count}")
        lines.extend(
            [
                (
                    "# HELP note_rag_http_request_duration_seconds_sum "
                    "Cumulative HTTP request duration."
                ),
                "# TYPE note_rag_http_request_duration_seconds_sum counter",
            ]
        )
        for (method, route), duration in sorted(durations.items()):
            labels = _labels(method=method, route=route)
            lines.append(
                "note_rag_http_request_duration_seconds_sum"
                f"{{{labels}}} {duration:.6f}"
            )
        return "\n".join(lines) + "\n"


def _labels(**values: str) -> str:
    return ",".join(
        f'{key}="{_escape_label(value)}"'
        for key, value in values.items()
    )


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')

"""Persistent, TTL-bound caches for exact RAG queries."""

import hashlib
import json
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any


class PersistentCache:
    """Small SQLite cache safe to share across application restarts and threads."""

    def __init__(self, path: Path, *, ttl_seconds: int = 3600) -> None:
        if ttl_seconds <= 0:
            raise ValueError("cache TTL must be greater than zero")
        self.path = path
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._stats: Counter[tuple[str, str]] = Counter()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    namespace TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    PRIMARY KEY (namespace, cache_key)
                )
                """
            )

    def get(self, namespace: str, key: str) -> Any | None:
        now = time.time()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT value_json, expires_at
                FROM cache_entries
                WHERE namespace = ? AND cache_key = ?
                """,
                (namespace, key),
            ).fetchone()
            if row is None:
                self._stats[(namespace, "miss")] += 1
                return None
            if float(row[1]) <= now:
                connection.execute(
                    """
                    DELETE FROM cache_entries
                    WHERE namespace = ? AND cache_key = ?
                    """,
                    (namespace, key),
                )
                self._stats[(namespace, "expired")] += 1
                self._stats[(namespace, "miss")] += 1
                return None
            self._stats[(namespace, "hit")] += 1
            return json.loads(str(row[0]))

    def set(self, namespace: str, key: str, value: Any) -> None:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        expires_at = time.time() + self.ttl_seconds
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO cache_entries (
                    namespace, cache_key, value_json, expires_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(namespace, cache_key) DO UPDATE SET
                    value_json = excluded.value_json,
                    expires_at = excluded.expires_at
                """,
                (namespace, key, encoded, expires_at),
            )
            self._stats[(namespace, "write")] += 1

    def clear(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM cache_entries")

    def stats(self) -> dict[tuple[str, str], int]:
        with self._lock:
            return dict(self._stats)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)


def cache_key(payload: dict[str, Any]) -> str:
    """Create a stable, non-sensitive key from a versioned JSON payload."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()

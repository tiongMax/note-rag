import time
from pathlib import Path

from note_rag.cache import PersistentCache


def test_cache_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite3"
    PersistentCache(path, ttl_seconds=60).set(
        "query_embedding", "key", [1.0, 2.0]
    )
    assert PersistentCache(path, ttl_seconds=60).get(
        "query_embedding", "key"
    ) == [1.0, 2.0]


def test_cache_expires_entries(tmp_path: Path) -> None:
    cache = PersistentCache(tmp_path / "cache.sqlite3", ttl_seconds=1)
    cache.set("retrieval", "key", {"value": 1})
    with cache._connect() as connection:
        connection.execute(
            "UPDATE cache_entries SET expires_at = ?",
            (time.time() - 1,),
        )
    assert cache.get("retrieval", "key") is None
    assert cache.stats()[("retrieval", "expired")] == 1

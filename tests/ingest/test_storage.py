import hashlib
from pathlib import Path

from note_rag.ingest import LocalFileStorage


def test_stores_content_by_hash_without_trusting_filename(tmp_path: Path) -> None:
    content = b"same bytes"
    content_hash = hashlib.sha256(content).hexdigest()
    storage = LocalFileStorage(tmp_path)

    first = storage.save(
        content,
        filename="../../notes.TXT",
        content_hash=content_hash,
    )
    second = storage.save(
        content,
        filename="renamed.txt",
        content_hash=content_hash,
    )

    assert first.path == second.path
    assert first.path.parent == tmp_path.resolve() / content_hash[:2]
    assert first.path.name == f"{content_hash}.txt"
    assert first.path.read_bytes() == content

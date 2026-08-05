"""Content-addressed local file storage."""

import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class StoredFile:
    uri: str
    path: Path


class LocalFileStorage:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def save(
        self,
        content: bytes,
        *,
        filename: str,
        content_hash: str,
    ) -> StoredFile:
        suffix = Path(filename).suffix.lower()
        directory = self.root / content_hash[:2]
        path = directory / f"{content_hash}{suffix}"
        directory.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary_path = directory / f".{path.name}.{uuid.uuid4().hex}.tmp"
            try:
                temporary_path.write_bytes(content)
                temporary_path.replace(path)
            finally:
                temporary_path.unlink(missing_ok=True)
        return StoredFile(uri=path.as_uri(), path=path)

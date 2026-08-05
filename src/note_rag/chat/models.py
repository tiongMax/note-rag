"""Framework-independent chat, prompt, citation, and streaming contracts."""

import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ChatTurn:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class Citation:
    citation_id: int
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    position: int
    source_metadata: dict[str, Any]

    def for_storage(self) -> dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "filename": self.filename,
            "position": self.position,
            "source_metadata": self.source_metadata,
        }


@dataclass(frozen=True, slots=True)
class ChatResult:
    conversation_id: uuid.UUID
    message_id: uuid.UUID
    answer: str
    citations: list[Citation]
    model_name: str


@dataclass(frozen=True, slots=True)
class ChatStreamEvent:
    event: str
    data: dict[str, Any]

"""Framework-independent chat, prompt, citation, and streaming contracts."""

import uuid
from dataclasses import dataclass
from typing import Any

from note_rag.context import ContextPackage
from note_rag.guardrails import GuardrailDecision


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
class ChatGuardrailTrace:
    """Auditable local decisions applied to one completed chat request."""

    input: GuardrailDecision
    context: GuardrailDecision
    output: GuardrailDecision
    filtered_context_chunks: int
    lexical_groundedness_proxy: float | None
    lexical_relevance_proxy: float | None


@dataclass(frozen=True, slots=True)
class ChatMemoryTrace:
    """Auditable history selection for one generation prompt."""

    recent_message_count: int
    semantic_memory_count: int
    available_older_memory_count: int
    recent_tokens: int
    semantic_memory_tokens: int
    embedding_model: str | None
    retrieval_query_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ChatResult:
    conversation_id: uuid.UUID
    message_id: uuid.UUID
    answer: str
    citations: list[Citation]
    model_name: str
    generation_context: ContextPackage
    generation_prompt_sha256: str
    prompt_token_count: int
    prompt_token_budget: int
    guardrails: ChatGuardrailTrace | None = None
    memory: ChatMemoryTrace | None = None


@dataclass(frozen=True, slots=True)
class ChatStreamEvent:
    event: str
    data: dict[str, Any]

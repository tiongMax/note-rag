"""Grounded chat providers, prompts, persistence orchestration, and citations."""

from note_rag.chat.models import (
    ChatGuardrailTrace,
    ChatMemoryTrace,
    ChatResult,
    ChatStreamEvent,
    ChatTurn,
    Citation,
)
from note_rag.chat.prompts import (
    GROUNDED_SYSTEM_PROMPT,
    NO_RETRIEVED_CONTEXT,
    build_chat_turns,
    prompt_fixed_text,
    prompt_sha256,
    rendered_generation_context,
)
from note_rag.chat.providers import (
    ChatProvider,
    GeminiChatProvider,
)
from note_rag.chat.service import (
    ChatContextBuilder,
    ChatGenerationError,
    ChatOptions,
    ChatOutputLimitError,
    ChatOutputValidationError,
    ChatPromptLimitError,
    ChatService,
    GuardrailRejectionError,
)

__all__ = [
    "ChatOptions",
    "ChatContextBuilder",
    "ChatGenerationError",
    "ChatGuardrailTrace",
    "ChatMemoryTrace",
    "ChatOutputLimitError",
    "ChatOutputValidationError",
    "ChatProvider",
    "ChatPromptLimitError",
    "ChatResult",
    "ChatService",
    "ChatStreamEvent",
    "ChatTurn",
    "Citation",
    "GROUNDED_SYSTEM_PROMPT",
    "NO_RETRIEVED_CONTEXT",
    "GeminiChatProvider",
    "GuardrailRejectionError",
    "build_chat_turns",
    "prompt_sha256",
    "prompt_fixed_text",
    "rendered_generation_context",
]

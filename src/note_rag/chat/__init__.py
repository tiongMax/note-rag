"""Grounded chat providers, prompts, persistence orchestration, and citations."""

from note_rag.chat.models import (
    ChatResult,
    ChatStreamEvent,
    ChatTurn,
    Citation,
)
from note_rag.chat.prompts import (
    GROUNDED_SYSTEM_PROMPT,
    NO_RETRIEVED_CONTEXT,
    build_chat_turns,
    prompt_sha256,
    rendered_generation_context,
)
from note_rag.chat.providers import (
    ChatProvider,
    GeminiChatProvider,
)
from note_rag.chat.service import ChatContextBuilder, ChatOptions, ChatService

__all__ = [
    "ChatOptions",
    "ChatContextBuilder",
    "ChatProvider",
    "ChatResult",
    "ChatService",
    "ChatStreamEvent",
    "ChatTurn",
    "Citation",
    "GROUNDED_SYSTEM_PROMPT",
    "NO_RETRIEVED_CONTEXT",
    "GeminiChatProvider",
    "build_chat_turns",
    "prompt_sha256",
    "rendered_generation_context",
]

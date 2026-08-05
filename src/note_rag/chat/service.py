"""Grounded chat orchestration with persistence, history, and citations."""

import re
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from note_rag.chat.models import (
    ChatResult,
    ChatStreamEvent,
    ChatTurn,
    Citation,
)
from note_rag.chat.prompts import GROUNDED_SYSTEM_PROMPT, build_chat_turns
from note_rag.chat.providers import ChatProvider
from note_rag.chunking import RegexTokenCounter
from note_rag.context import ContextPackage
from note_rag.persistence import (
    ChatMessageRecord,
    ChatMessageRepository,
    ChatRole,
    Conversation,
    ConversationRepository,
    Database,
)
from note_rag.retrieval import SearchFilters, SearchMode

_CITATION_PATTERN = re.compile(r"\[(\d+)]")
_WHITESPACE = re.compile(r"\s+")


class ChatContextBuilder(Protocol):
    def build(
        self,
        query: str,
        *,
        mode: SearchMode = SearchMode.HYBRID,
        candidate_k: int = 20,
        max_chunks: int = 8,
        max_context_tokens: int = 1200,
        vector_weight: float = 0.7,
        rerank: bool = True,
        rerank_weight: float = 0.7,
        filters: SearchFilters | None = None,
    ) -> ContextPackage: ...


@dataclass(frozen=True, slots=True)
class ChatOptions:
    mode: SearchMode = SearchMode.HYBRID
    candidate_k: int = 20
    max_chunks: int = 8
    max_context_tokens: int = 1200
    vector_weight: float = 0.7
    rerank: bool = True
    rerank_weight: float = 0.7
    filters: SearchFilters | None = None


@dataclass(frozen=True, slots=True)
class _PreparedChat:
    conversation_id: uuid.UUID
    context: ContextPackage
    turns: list[ChatTurn]


class ChatService:
    def __init__(
        self,
        database: Database,
        context_builder: ChatContextBuilder,
        provider: ChatProvider,
        *,
        token_counter: RegexTokenCounter | None = None,
        history_max_messages: int = 20,
        history_max_tokens: int = 2000,
    ) -> None:
        if history_max_messages <= 0:
            raise ValueError("history_max_messages must be greater than zero")
        if history_max_tokens <= 0:
            raise ValueError("history_max_tokens must be greater than zero")
        self.database = database
        self.context_builder = context_builder
        self.provider = provider
        self.token_counter = token_counter or RegexTokenCounter()
        self.history_max_messages = history_max_messages
        self.history_max_tokens = history_max_tokens

    def ask(
        self,
        question: str,
        *,
        conversation_id: uuid.UUID | None = None,
        options: ChatOptions | None = None,
    ) -> ChatResult:
        prepared = self._prepare(question, conversation_id, options)
        answer = self.provider.generate(
            GROUNDED_SYSTEM_PROMPT,
            prepared.turns,
        ).strip()
        if not answer:
            raise RuntimeError("chat provider returned an empty answer")
        return self._finalize(prepared, answer)

    def stream(
        self,
        question: str,
        *,
        conversation_id: uuid.UUID | None = None,
        options: ChatOptions | None = None,
    ) -> Iterator[ChatStreamEvent]:
        prepared = self._prepare(question, conversation_id, options)
        available = self._available_citations(prepared.context)
        yield ChatStreamEvent(
            event="metadata",
            data={
                "conversation_id": str(prepared.conversation_id),
                "sources": [item.for_storage() for item in available],
                "model_name": self.provider.model_name,
            },
        )
        pieces = []
        for piece in self.provider.stream(
            GROUNDED_SYSTEM_PROMPT,
            prepared.turns,
        ):
            if not piece:
                continue
            pieces.append(piece)
            yield ChatStreamEvent(event="delta", data={"text": piece})
        answer = "".join(pieces).strip()
        if not answer:
            raise RuntimeError("chat provider returned an empty answer")
        result = self._finalize(prepared, answer)
        yield ChatStreamEvent(
            event="done",
            data={
                "conversation_id": str(result.conversation_id),
                "message_id": str(result.message_id),
                "citations": [
                    citation.for_storage() for citation in result.citations
                ],
            },
        )

    def _prepare(
        self,
        question: str,
        conversation_id: uuid.UUID | None,
        options: ChatOptions | None,
    ) -> _PreparedChat:
        question = question.strip()
        if not question:
            raise ValueError("question cannot be empty")
        resolved = options or ChatOptions()
        history, existing = self._load_history(conversation_id)
        context = self.context_builder.build(
            question,
            mode=resolved.mode,
            candidate_k=resolved.candidate_k,
            max_chunks=resolved.max_chunks,
            max_context_tokens=resolved.max_context_tokens,
            vector_weight=resolved.vector_weight,
            rerank=resolved.rerank,
            rerank_weight=resolved.rerank_weight,
            filters=resolved.filters,
        )
        with self.database.session() as session:
            conversations = ConversationRepository(session)
            conversation = (
                conversations.get(existing)
                if existing is not None
                else conversations.add(
                    Conversation(title=self._title(question))
                )
            )
            if conversation is None:
                raise LookupError("conversation not found")
            ChatMessageRepository(session).add(
                conversation,
                role=ChatRole.USER,
                content=question,
                token_count=self.token_counter.count(question),
                context_token_count=context.token_count,
            )
            resolved_id = conversation.id
        return _PreparedChat(
            conversation_id=resolved_id,
            context=context,
            turns=build_chat_turns(question, context, history),
        )

    def _load_history(
        self,
        conversation_id: uuid.UUID | None,
    ) -> tuple[list[ChatTurn], uuid.UUID | None]:
        if conversation_id is None:
            return [], None
        with self.database.session() as session:
            conversation = ConversationRepository(session).get(conversation_id)
            if conversation is None:
                raise LookupError("conversation not found")
            messages = ChatMessageRepository(session).list_for_conversation(
                conversation_id
            )
        selected: list[ChatMessageRecord] = []
        used_tokens = 0
        for message in reversed(messages[-self.history_max_messages :]):
            if used_tokens + message.token_count > self.history_max_tokens:
                break
            selected.append(message)
            used_tokens += message.token_count
        selected.reverse()
        return [
            ChatTurn(role=message.role.value, content=message.content)
            for message in selected
        ], conversation_id

    def _finalize(
        self,
        prepared: _PreparedChat,
        answer: str,
    ) -> ChatResult:
        citations = self._extract_citations(answer, prepared.context)
        with self.database.session() as session:
            conversation = ConversationRepository(session).get(
                prepared.conversation_id
            )
            if conversation is None:
                raise LookupError("conversation not found")
            message = ChatMessageRepository(session).add(
                conversation,
                role=ChatRole.ASSISTANT,
                content=answer,
                token_count=self.token_counter.count(answer),
                citations=[item.for_storage() for item in citations],
                context_token_count=prepared.context.token_count,
                model_name=self.provider.model_name,
            )
        return ChatResult(
            conversation_id=prepared.conversation_id,
            message_id=message.id,
            answer=answer,
            citations=citations,
            model_name=self.provider.model_name,
        )

    @staticmethod
    def _available_citations(context: ContextPackage) -> list[Citation]:
        return [
            Citation(
                citation_id=chunk.citation_id,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                position=chunk.position,
                source_metadata=chunk.source_metadata,
            )
            for chunk in context.chunks
        ]

    def _extract_citations(
        self,
        answer: str,
        context: ContextPackage,
    ) -> list[Citation]:
        available = {
            citation.citation_id: citation
            for citation in self._available_citations(context)
        }
        referenced = {
            int(match.group(1))
            for match in _CITATION_PATTERN.finditer(answer)
        }
        return [
            available[citation_id]
            for citation_id in sorted(referenced)
            if citation_id in available
        ]

    @staticmethod
    def _title(question: str) -> str:
        normalized = _WHITESPACE.sub(" ", question).strip()
        return normalized[:80]

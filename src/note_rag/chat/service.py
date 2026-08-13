"""Grounded chat orchestration with persistence, history, and citations."""

import re
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol

from note_rag.chat.budget import PromptBudget, trim_history_to_budget
from note_rag.chat.models import (
    ChatGuardrailTrace,
    ChatResult,
    ChatStreamEvent,
    ChatTurn,
    Citation,
)
from note_rag.chat.prompts import (
    GROUNDED_SYSTEM_PROMPT,
    build_chat_turns,
    prompt_fixed_text,
    prompt_sha256,
    rendered_generation_context,
)
from note_rag.chat.providers import ChatProvider
from note_rag.chunking import RegexTokenCounter
from note_rag.context import ContextPackage
from note_rag.guardrails import (
    UNSUPPORTED_OUTPUT_RESPONSE,
    GuardrailAction,
    GuardrailDecision,
    GuardrailService,
    GuardrailStage,
    OutputGuardResult,
)
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
    new_conversation: bool
    question: str
    context: ContextPackage
    turns: list[ChatTurn]
    prompt_budget: PromptBudget
    input_decision: GuardrailDecision | None
    context_decision: GuardrailDecision | None
    filtered_context_chunks: int


class GuardrailRejectionError(ValueError):
    """Raised before retrieval when input safety controls reject a request."""

    def __init__(self, decision: GuardrailDecision) -> None:
        self.decision = decision
        super().__init__("request rejected by input safety controls")


class ChatGenerationError(RuntimeError):
    """A provider failed without exposing provider details to callers."""

    def __init__(self) -> None:
        super().__init__("chat provider generation failed")


class ChatOutputValidationError(RuntimeError):
    """Generated output could not be validated safely."""

    def __init__(self) -> None:
        super().__init__("chat output validation failed")


class ChatOutputLimitError(RuntimeError):
    """Generated output exceeded the bounded accumulation size."""

    def __init__(self) -> None:
        super().__init__("chat provider output exceeded the service limit")


class ChatPromptLimitError(ValueError):
    """The assembled prompt exceeded its character budget."""

    def __init__(self) -> None:
        super().__init__("assembled chat prompt exceeds the global character budget")


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
        prompt_max_tokens: int = 4096,
        prompt_reserve_tokens: int = 128,
        prompt_max_characters: int | None = None,
        output_max_characters: int | None = None,
        guardrails: GuardrailService | None = None,
        decision_observer: Callable[[GuardrailDecision], None] | None = None,
    ) -> None:
        if history_max_messages <= 0:
            raise ValueError("history_max_messages must be greater than zero")
        if history_max_tokens <= 0:
            raise ValueError("history_max_tokens must be greater than zero")
        if prompt_max_tokens <= 0:
            raise ValueError("prompt_max_tokens must be greater than zero")
        if not 0 < prompt_reserve_tokens < prompt_max_tokens:
            raise ValueError(
                "prompt_reserve_tokens must be positive and smaller than "
                "prompt_max_tokens"
            )
        resolved_prompt_max_characters = (
            prompt_max_tokens * 8
            if prompt_max_characters is None
            else prompt_max_characters
        )
        if resolved_prompt_max_characters <= 0:
            raise ValueError("prompt_max_characters must be greater than zero")
        resolved_output_max_characters = (
            (guardrails.output_max_tokens if guardrails is not None else 1024) * 16
            if output_max_characters is None
            else output_max_characters
        )
        if resolved_output_max_characters <= 0:
            raise ValueError("output_max_characters must be greater than zero")
        self.database = database
        self.context_builder = context_builder
        self.provider = provider
        self.token_counter = token_counter or RegexTokenCounter()
        self.history_max_messages = history_max_messages
        self.history_max_tokens = history_max_tokens
        self.prompt_max_tokens = prompt_max_tokens
        self.prompt_reserve_tokens = prompt_reserve_tokens
        self.prompt_max_characters = resolved_prompt_max_characters
        self.output_max_characters = resolved_output_max_characters
        self.guardrails = guardrails
        self.decision_observer = decision_observer

    def ask(
        self,
        question: str,
        *,
        conversation_id: uuid.UUID | None = None,
        options: ChatOptions | None = None,
    ) -> ChatResult:
        prepared = self._prepare(question, conversation_id, options)
        if self.guardrails is not None and not prepared.context.chunks:
            answer = UNSUPPORTED_OUTPUT_RESPONSE
        else:
            try:
                answer = self.provider.generate(
                    GROUNDED_SYSTEM_PROMPT,
                    prepared.turns,
                )
            except Exception as error:
                raise ChatGenerationError() from error
        self._enforce_output_character_limit(answer)
        return self._finalize(prepared, answer)

    def stream(
        self,
        question: str,
        *,
        conversation_id: uuid.UUID | None = None,
        options: ChatOptions | None = None,
    ) -> Iterator[ChatStreamEvent]:
        prepared = self._prepare(question, conversation_id, options)
        return self._stream_prepared(prepared)

    def _stream_prepared(
        self,
        prepared: _PreparedChat,
    ) -> Iterator[ChatStreamEvent]:
        pieces: list[str] = []
        accumulated_characters = 0
        if self.guardrails is not None and not prepared.context.chunks:
            pieces.append(UNSUPPORTED_OUTPUT_RESPONSE)
        else:
            try:
                provider_pieces = self.provider.stream(
                    GROUNDED_SYSTEM_PROMPT,
                    prepared.turns,
                )
                for piece in provider_pieces:
                    if not piece:
                        continue
                    accumulated_characters += len(piece)
                    if accumulated_characters > self.output_max_characters:
                        raise ChatOutputLimitError()
                    pieces.append(piece)
                    if (
                        self.guardrails is not None
                        and self.token_counter.count("".join(pieces))
                        > self.guardrails.output_max_tokens
                    ):
                        break
            except ChatOutputLimitError:
                raise
            except Exception as error:
                raise ChatGenerationError() from error
        result = self._finalize(prepared, "".join(pieces))
        yield ChatStreamEvent(
            event="metadata",
            data={
                "conversation_id": str(prepared.conversation_id),
                "sources": [
                    citation.for_storage() for citation in result.citations
                ],
                "model_name": self.provider.model_name,
                "guardrails": self._trace_for_storage(result.guardrails),
            },
        )
        original_answer = "".join(pieces).strip()
        if result.answer == original_answer:
            for piece in pieces:
                yield ChatStreamEvent(event="delta", data={"text": piece})
        else:
            yield ChatStreamEvent(event="delta", data={"text": result.answer})
        yield ChatStreamEvent(
            event="done",
            data={
                "conversation_id": str(result.conversation_id),
                "message_id": str(result.message_id),
                "citations": [
                    citation.for_storage() for citation in result.citations
                ],
                "guardrails": self._trace_for_storage(result.guardrails),
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
        fixed_prompt_characters = len(GROUNDED_SYSTEM_PROMPT) + len(
            prompt_fixed_text(question)
        )
        if fixed_prompt_characters > self.prompt_max_characters:
            raise ChatPromptLimitError()
        input_decision = None
        if self.guardrails is not None:
            input_decision = self.guardrails.inspect_input(question)
            self._observe(input_decision)
            if input_decision.action is GuardrailAction.BLOCK:
                raise GuardrailRejectionError(input_decision)
        resolved = options or ChatOptions()
        history, existing = self._load_history(conversation_id)
        history, context_budget, _ = trim_history_to_budget(
            history,
            question=prompt_fixed_text(question),
            system_instruction=GROUNDED_SYSTEM_PROMPT,
            prompt_max_tokens=self.prompt_max_tokens,
            prompt_reserve_tokens=self.prompt_reserve_tokens,
            context_max_tokens=resolved.max_context_tokens,
            token_counter=self.token_counter,
        )
        context = self.context_builder.build(
            question,
            mode=resolved.mode,
            candidate_k=resolved.candidate_k,
            max_chunks=resolved.max_chunks,
            max_context_tokens=context_budget,
            vector_weight=resolved.vector_weight,
            rerank=resolved.rerank,
            rerank_weight=resolved.rerank_weight,
            filters=resolved.filters,
        )
        context_decision = None
        filtered_context_chunks = 0
        if self.guardrails is not None:
            guarded_context = self.guardrails.sanitize_context(context)
            context = guarded_context.context
            context_decision = guarded_context.decision
            filtered_context_chunks = guarded_context.filtered_chunk_count
            self._observe(context_decision)
        turns = build_chat_turns(question, context, history)
        assembled_characters = len(GROUNDED_SYSTEM_PROMPT) + sum(
            len(turn.content) for turn in turns
        )
        if assembled_characters > self.prompt_max_characters:
            raise ChatPromptLimitError()
        assembled_tokens = self.token_counter.count(GROUNDED_SYSTEM_PROMPT) + sum(
            self.token_counter.count(turn.content) for turn in turns
        )
        used_with_reserve = assembled_tokens + self.prompt_reserve_tokens
        if used_with_reserve > self.prompt_max_tokens:
            raise ValueError("assembled chat prompt exceeds the global token budget")
        return _PreparedChat(
            conversation_id=existing or uuid.uuid4(),
            new_conversation=existing is None,
            question=question,
            context=context,
            turns=turns,
            prompt_budget=PromptBudget(
                maximum=self.prompt_max_tokens,
                reserve=self.prompt_reserve_tokens,
                used=used_with_reserve,
                history_tokens=sum(
                    self.token_counter.count(turn.content) for turn in history
                ),
                context_tokens=context.token_count,
            ),
            input_decision=input_decision,
            context_decision=context_decision,
            filtered_context_chunks=filtered_context_chunks,
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
        try:
            guarded_output = self._guard_output(prepared, answer)
        except Exception as error:
            raise ChatOutputValidationError() from error
        answer = guarded_output.answer.strip()
        if not answer:
            raise ChatOutputValidationError()
        try:
            citations = self._extract_citations(answer, prepared.context)
        except Exception as error:
            raise ChatOutputValidationError() from error
        generation_context = self._generation_context(prepared.context)
        generation_prompt_sha256 = prompt_sha256(
            GROUNDED_SYSTEM_PROMPT,
            prepared.turns,
        )
        guardrail_trace = self._guardrail_trace(prepared, guarded_output)
        with self.database.session() as session:
            conversations = ConversationRepository(session)
            conversation = (
                conversations.add(
                    Conversation(
                        id=prepared.conversation_id,
                        title=self._title(prepared.question),
                    )
                )
                if prepared.new_conversation
                else conversations.get(prepared.conversation_id)
            )
            if conversation is None:
                raise LookupError("conversation not found")
            messages = ChatMessageRepository(session)
            messages.add(
                conversation,
                role=ChatRole.USER,
                content=prepared.question,
                token_count=self.token_counter.count(prepared.question),
                context_token_count=prepared.context.token_count,
            )
            message = messages.add(
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
            generation_context=generation_context,
            generation_prompt_sha256=generation_prompt_sha256,
            prompt_token_count=(
                prepared.prompt_budget.used - prepared.prompt_budget.reserve
            ),
            prompt_token_budget=prepared.prompt_budget.maximum,
            guardrails=guardrail_trace,
        )

    def _enforce_output_character_limit(self, answer: str) -> None:
        if len(answer) > self.output_max_characters:
            raise ChatOutputLimitError()

    def _guard_output(
        self,
        prepared: _PreparedChat,
        answer: str,
    ) -> OutputGuardResult:
        if self.guardrails is None:
            return OutputGuardResult(
                answer=answer,
                decision=GuardrailDecision(
                    stage=GuardrailStage.OUTPUT,
                    action=GuardrailAction.ALLOW,
                    reason_codes=(),
                    latency_ms=0.0,
                ),
                lexical_groundedness_proxy=None,
                lexical_relevance_proxy=None,
            )
        guarded = self.guardrails.validate_output(
            question=prepared.question,
            answer=answer,
            context=prepared.context,
        )
        self._observe(guarded.decision)
        return guarded

    def _guardrail_trace(
        self,
        prepared: _PreparedChat,
        output: OutputGuardResult,
    ) -> ChatGuardrailTrace | None:
        if (
            self.guardrails is None
            or prepared.input_decision is None
            or prepared.context_decision is None
        ):
            return None
        return ChatGuardrailTrace(
            input=prepared.input_decision,
            context=prepared.context_decision,
            output=output.decision,
            filtered_context_chunks=prepared.filtered_context_chunks,
            lexical_groundedness_proxy=output.lexical_groundedness_proxy,
            lexical_relevance_proxy=output.lexical_relevance_proxy,
        )

    def _observe(self, decision: GuardrailDecision) -> None:
        if self.decision_observer is not None:
            self.decision_observer(decision)

    @staticmethod
    def _trace_for_storage(
        trace: ChatGuardrailTrace | None,
    ) -> dict[str, object] | None:
        if trace is None:
            return None
        return {
            "input": {
                "stage": trace.input.stage.value,
                "action": trace.input.action.value,
                "reason_codes": list(trace.input.reason_codes),
                "latency_ms": trace.input.latency_ms,
            },
            "context": {
                "stage": trace.context.stage.value,
                "action": trace.context.action.value,
                "reason_codes": list(trace.context.reason_codes),
                "latency_ms": trace.context.latency_ms,
            },
            "output": {
                "stage": trace.output.stage.value,
                "action": trace.output.action.value,
                "reason_codes": list(trace.output.reason_codes),
                "latency_ms": trace.output.latency_ms,
            },
            "filtered_context_chunks": trace.filtered_context_chunks,
            "lexical_groundedness_proxy": trace.lexical_groundedness_proxy,
            "lexical_relevance_proxy": trace.lexical_relevance_proxy,
        }

    @staticmethod
    def _generation_context(context: ContextPackage) -> ContextPackage:
        """Expose the exact evidence text used by prompt construction."""

        rendered = rendered_generation_context(context)
        if rendered == context.context:
            return context
        return ContextPackage(
            query=context.query,
            mode=context.mode,
            context=rendered,
            chunks=context.chunks,
            token_count=context.token_count,
            token_budget=context.token_budget,
            candidates_considered=context.candidates_considered,
            duplicates_removed=context.duplicates_removed,
            truncated=context.truncated,
            reranker_model=context.reranker_model,
            corpus_version=context.corpus_version,
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

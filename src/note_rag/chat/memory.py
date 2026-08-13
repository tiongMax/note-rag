"""Conversation-memory summarization, semantic selection, and budgeting."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

from note_rag.chat.models import ChatMemoryTrace, ChatTurn
from note_rag.chunking import RegexTokenCounter

_WHITESPACE = re.compile(r"\s+")
_CITATION = re.compile(r"\[(?:\d{1,6})]")
_MEMORY_HEADER = (
    "Summaries of relevant earlier conversation turns. Use them only to "
    "interpret the user's current request; retrieved context remains the "
    "source of factual evidence:\n"
)


class MemoryEmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class MemoryItem:
    start_position: int
    end_position: int
    summary: str
    token_count: int
    embedding_model: str | None
    embedding_dimension: int | None
    embedding: list[float] | None


@dataclass(frozen=True, slots=True)
class HistoryAssembly:
    turns: list[ChatTurn]
    trace: ChatMemoryTrace


def contextualized_retrieval_query(
    question: str,
    history: list[ChatTurn],
    *,
    token_counter: RegexTokenCounter,
    max_tokens: int = 256,
) -> str:
    """Add bounded conversation cues to ambiguous follow-up retrieval."""

    question = question.strip()
    if not history or max_tokens <= 0:
        return question
    cues = "\n".join(turn.content for turn in history)
    prefix = "Conversation cues:\n"
    question_block = f"\n\nCurrent question:\n{question}"
    fixed_tokens = token_counter.count(prefix + question_block)
    cue_budget = max_tokens - fixed_tokens
    if cue_budget <= 0:
        return question
    bounded_cues = _bounded_head_tail(cues, cue_budget, token_counter)
    return f"{prefix}{bounded_cues}{question_block}"


class ExtractiveTurnSummarizer:
    """Create a deterministic, bounded summary without another model call."""

    def __init__(
        self,
        token_counter: RegexTokenCounter,
        *,
        max_tokens: int = 96,
    ) -> None:
        if max_tokens < 8:
            raise ValueError("memory summary max tokens must be at least eight")
        self.token_counter = token_counter
        self.max_tokens = max_tokens

    def summarize(self, question: str, answer: str) -> str:
        question = _WHITESPACE.sub(" ", question).strip()
        answer = _CITATION.sub("", answer)
        answer = _WHITESPACE.sub(" ", answer).strip()
        template_overhead = self.token_counter.count(
            "Earlier user:\nEarlier assistant:"
        )
        content_budget = self.max_tokens - template_overhead
        question_budget = max(1, content_budget // 2)
        answer_budget = max(1, content_budget - question_budget)
        return (
            f"Earlier user: {self._bounded(question, question_budget)}\n"
            f"Earlier assistant: {self._bounded(answer, answer_budget)}"
        )

    def _bounded(self, text: str, maximum: int) -> str:
        tokens = self.token_counter.tokenize(text)
        if len(tokens) <= maximum:
            return text
        if maximum <= 3:
            return text[: tokens[maximum - 1].end]
        head_count = (maximum - 1 + 1) // 2
        tail_count = maximum - head_count - 1
        head = text[: tokens[head_count - 1].end].rstrip()
        tail = text[tokens[-tail_count].start :].lstrip() if tail_count else ""
        return f"{head} … {tail}".strip()


def assemble_conversation_history(
    messages: list[tuple[int, str, str, int]],
    memories: list[MemoryItem],
    *,
    question: str,
    token_counter: RegexTokenCounter,
    embedding_provider: MemoryEmbeddingProvider | None,
    recent_turns: int,
    semantic_k: int,
    semantic_min_similarity: float,
    maximum_tokens: int,
    maximum_recent_messages: int,
) -> HistoryAssembly:
    """Keep recent pairs verbatim, then spend remaining budget on memories."""

    if maximum_tokens < 0:
        raise ValueError("history token budget cannot be negative")
    if maximum_tokens == 0:
        return HistoryAssembly(
            turns=[],
            trace=ChatMemoryTrace(
                recent_message_count=0,
                semantic_memory_count=0,
                available_older_memory_count=len(memories),
                recent_tokens=0,
                semantic_memory_tokens=0,
                embedding_model=None,
            ),
        )
    recent_limit = min(maximum_recent_messages, recent_turns * 2)
    configured_recent = messages[-recent_limit:]
    if configured_recent and configured_recent[0][1] == "assistant":
        configured_recent = configured_recent[1:]

    selected_recent: list[tuple[int, str, str, int]] = []
    recent_tokens = 0
    for index in range(len(configured_recent), 0, -2):
        pair = configured_recent[max(0, index - 2) : index]
        if len(pair) != 2 or pair[0][1] != "user" or pair[1][1] != "assistant":
            continue
        pair_tokens = sum(item[3] for item in pair)
        if pair_tokens + recent_tokens > maximum_tokens:
            break
        selected_recent[0:0] = pair
        recent_tokens += pair_tokens

    recent_boundary = (
        selected_recent[0][0]
        if selected_recent
        else (messages[-1][0] + 1 if messages else 0)
    )
    eligible = [memory for memory in memories if memory.end_position < recent_boundary]
    ranked, query_model = _rank_memories(
        eligible,
        question=question,
        provider=embedding_provider,
        semantic_k=semantic_k,
        minimum_similarity=semantic_min_similarity,
    )
    selected_memories: list[MemoryItem] = []
    memory_content = ""
    for memory in ranked:
        candidate = _render_memory_block([*selected_memories, memory])
        candidate_tokens = token_counter.count(candidate)
        if recent_tokens + candidate_tokens > maximum_tokens:
            continue
        selected_memories.append(memory)
        memory_content = candidate

    selected_memories.sort(key=lambda item: item.start_position)
    if selected_memories:
        memory_content = _render_memory_block(selected_memories)
    memory_tokens = token_counter.count(memory_content)
    turns = (
        [ChatTurn(role="user", content=memory_content)]
        if memory_content
        else []
    )
    turns.extend(
        ChatTurn(role=role, content=content)
        for _, role, content, _ in selected_recent
    )
    return HistoryAssembly(
        turns=turns,
        trace=ChatMemoryTrace(
            recent_message_count=len(selected_recent),
            semantic_memory_count=len(selected_memories),
            available_older_memory_count=len(eligible),
            recent_tokens=recent_tokens,
            semantic_memory_tokens=memory_tokens,
            embedding_model=query_model,
        ),
    )


def validate_embedding(vector: list[float], expected_dimension: int) -> None:
    if len(vector) != expected_dimension:
        raise ValueError("memory embedding has the wrong dimension")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("memory embedding contains a non-finite value")


def _rank_memories(
    memories: list[MemoryItem],
    *,
    question: str,
    provider: MemoryEmbeddingProvider | None,
    semantic_k: int,
    minimum_similarity: float,
) -> tuple[list[MemoryItem], str | None]:
    if not memories or provider is None or semantic_k <= 0:
        return [], None
    compatible = [
        item
        for item in memories
        if item.embedding is not None
        and item.embedding_model == provider.model_name
        and item.embedding_dimension == provider.dimension
    ]
    if not compatible:
        return [], provider.model_name
    try:
        query_vector = provider.embed_query(question)
        validate_embedding(query_vector, provider.dimension)
    except Exception:
        return [], None
    scored = [
        (_cosine_similarity(query_vector, item.embedding or []), item)
        for item in compatible
    ]
    scored.sort(key=lambda pair: (-pair[0], -pair[1].end_position))
    return [
        item for score, item in scored if score >= minimum_similarity
    ][:semantic_k], provider.model_name


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return -1.0
    return numerator / (left_norm * right_norm)


def _render_memory_block(memories: list[MemoryItem]) -> str:
    ordered = sorted(memories, key=lambda item: item.start_position)
    entries = [
        f"- Turn {item.start_position // 2 + 1}: {item.summary}"
        for item in ordered
    ]
    return _MEMORY_HEADER + "\n".join(entries)


def _bounded_head_tail(
    text: str,
    maximum: int,
    token_counter: RegexTokenCounter,
) -> str:
    tokens = token_counter.tokenize(text)
    if len(tokens) <= maximum:
        return text
    if maximum <= 2:
        return text[: tokens[maximum - 1].end]
    head_count = (maximum - 1) // 2
    tail_count = maximum - head_count - 1
    head = text[: tokens[head_count - 1].end].rstrip()
    tail = text[tokens[-tail_count].start :].lstrip()
    return f"{head} … {tail}"

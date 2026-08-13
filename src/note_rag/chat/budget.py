"""Deterministic application-token budgeting for assembled chat prompts."""

from __future__ import annotations

from dataclasses import dataclass

from note_rag.chat.models import ChatTurn
from note_rag.chunking import RegexTokenCounter


@dataclass(frozen=True, slots=True)
class PromptBudget:
    maximum: int
    reserve: int
    used: int
    history_tokens: int
    context_tokens: int


def trim_history_to_budget(
    history: list[ChatTurn],
    *,
    question: str,
    system_instruction: str,
    prompt_max_tokens: int,
    prompt_reserve_tokens: int,
    context_max_tokens: int,
    token_counter: RegexTokenCounter,
) -> tuple[list[ChatTurn], int, PromptBudget]:
    """Trim oldest turns, then return the remaining context-token budget."""

    fixed_tokens = token_counter.count(system_instruction) + token_counter.count(
        question
    )
    usable = prompt_max_tokens - prompt_reserve_tokens - fixed_tokens
    if usable <= 0:
        raise ValueError("question and prompt template exceed the chat prompt budget")
    selected = list(history)
    history_tokens = sum(token_counter.count(turn.content) for turn in selected)
    reserved_context_tokens = min(context_max_tokens, max(1, usable // 2))
    while (
        selected
        and history_tokens + reserved_context_tokens > usable
    ):
        removed = selected.pop(0)
        history_tokens -= token_counter.count(removed.content)
    context_budget = min(context_max_tokens, max(1, usable - history_tokens))
    used = fixed_tokens + history_tokens + context_budget + prompt_reserve_tokens
    return selected, context_budget, PromptBudget(
        maximum=prompt_max_tokens,
        reserve=prompt_reserve_tokens,
        used=used,
        history_tokens=history_tokens,
        context_tokens=context_budget,
    )

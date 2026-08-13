import pytest

from note_rag.chat.budget import trim_history_to_budget
from note_rag.chat.models import ChatTurn
from note_rag.chunking import RegexTokenCounter


def test_trims_oldest_history_then_shrinks_context_budget() -> None:
    counter = RegexTokenCounter()
    history = [
        ChatTurn(role="user", content="old one two three"),
        ChatTurn(role="assistant", content="recent four five"),
    ]

    selected, context_budget, budget = trim_history_to_budget(
        history,
        question="question",
        system_instruction="system",
        prompt_max_tokens=10,
        prompt_reserve_tokens=2,
        context_max_tokens=8,
        token_counter=counter,
    )

    assert selected == history[1:]
    assert context_budget == 3
    assert budget.used <= budget.maximum


def test_rejects_fixed_prompt_larger_than_global_budget() -> None:
    with pytest.raises(ValueError, match="exceed"):
        trim_history_to_budget(
            [],
            question="one two three",
            system_instruction="four five",
            prompt_max_tokens=5,
            prompt_reserve_tokens=1,
            context_max_tokens=1,
            token_counter=RegexTokenCounter(),
        )

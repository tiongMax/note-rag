"""Deterministic quality signals for generated educational material."""

import re
import statistics
from collections.abc import Sequence
from typing import Any

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(value: str) -> set[str]:
    return set(_TOKEN.findall(value.casefold()))


def citation_support(answer: str, source_passages: Sequence[str]) -> float:
    """Estimate whether answer concepts appear in at least one cited passage."""

    answer_tokens = _tokens(answer)
    if not answer_tokens or not source_passages:
        return 0.0
    source_tokens = _tokens(" ".join(source_passages))
    return len(answer_tokens & source_tokens) / len(answer_tokens)


def duplicate_question_rate(prompts: Sequence[str], *, threshold: float = 0.8) -> float:
    if len(prompts) < 2:
        return 0.0
    duplicate_indexes: set[int] = set()
    tokenized = [_tokens(prompt) for prompt in prompts]
    for right in range(1, len(tokenized)):
        for left in range(right):
            union = tokenized[left] | tokenized[right]
            similarity = (
                len(tokenized[left] & tokenized[right]) / len(union) if union else 1
            )
            if similarity >= threshold:
                duplicate_indexes.add(right)
                break
    return len(duplicate_indexes) / len(prompts)


def multiple_choice_quality(
    answer: str, options: Sequence[dict[str, Any]]
) -> dict[str, float]:
    correct = [option for option in options if option.get("correct") is True]
    ambiguity = float(len(correct) != 1)
    distractors = [
        str(option.get("text", "")).strip()
        for option in options
        if not option.get("correct")
    ]
    normalized_answer = " ".join(sorted(_tokens(answer)))
    valid = {
        " ".join(sorted(_tokens(distractor)))
        for distractor in distractors
        if distractor and " ".join(sorted(_tokens(distractor))) != normalized_answer
    }
    quality = len(valid) / len(distractors) if distractors else 0.0
    return {"answer_ambiguity": ambiguity, "distractor_quality": quality}


def difficulty_consistency(difficulties: Sequence[int]) -> float:
    """Return a stable 0..1 signal; large unexplained jumps reduce consistency."""

    if len(difficulties) < 2:
        return 1.0
    jumps = [
        abs(right - left) / 4 for left, right in zip(difficulties, difficulties[1:])
    ]
    return max(0.0, 1 - statistics.fmean(jumps))


def evaluate_learning_materials(items: Sequence[dict[str, Any]]) -> dict[str, float]:
    citations = [
        citation_support(str(item.get("answer", "")), item.get("sources", []))
        for item in items
    ]
    multiple_choice = [
        multiple_choice_quality(str(item.get("answer", "")), item.get("options", []))
        for item in items
        if item.get("item_type") == "multiple_choice"
    ]
    objectives = {
        str(item.get("objective_id")) for item in items if item.get("objective_id")
    }
    covered = {
        str(item.get("objective_id"))
        for item in items
        if item.get("objective_id") and item.get("approval_status") == "approved"
    }
    return {
        "citation_support": statistics.fmean(citations) if citations else 0.0,
        "duplicate_question_rate": duplicate_question_rate(
            [str(item.get("prompt", "")) for item in items]
        ),
        "answer_ambiguity": statistics.fmean(
            result["answer_ambiguity"] for result in multiple_choice
        )
        if multiple_choice
        else 0.0,
        "distractor_quality": statistics.fmean(
            result["distractor_quality"] for result in multiple_choice
        )
        if multiple_choice
        else 0.0,
        "difficulty_consistency": difficulty_consistency(
            [int(item.get("difficulty", 3)) for item in items]
        ),
        "objective_coverage": len(covered) / len(objectives) if objectives else 0.0,
    }

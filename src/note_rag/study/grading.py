"""Deterministic, inspectable grading for the first study workflow."""

import re
from dataclasses import dataclass

from note_rag.persistence import StudyItem, StudyItemType

_WORD = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


def _normalized(value: str) -> str:
    return " ".join(_WORD.findall(value.casefold()))


def _concepts(value: str) -> set[str]:
    return {word for word in _WORD.findall(value.casefold()) if word not in _STOP_WORDS}


@dataclass(frozen=True, slots=True)
class GradeResult:
    score: float
    correct: bool
    correct_concepts: list[str]
    missing_concepts: list[str]
    mistaken_concepts: list[str]
    rationale: str
    supporting_chunk_ids: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "correct_concepts": self.correct_concepts,
            "missing_concepts": self.missing_concepts,
            "mistaken_concepts": self.mistaken_concepts,
            "rationale": self.rationale,
            "supporting_chunk_ids": self.supporting_chunk_ids,
        }


def grade_answer(item: StudyItem, submitted_answer: str) -> GradeResult:
    """Return a structured grade whose evidence points to stored source chunks."""

    submitted = _normalized(submitted_answer)
    expected = _normalized(item.answer)
    source_ids = [str(source.chunk_id) for source in item.sources]

    if item.item_type is StudyItemType.MULTIPLE_CHOICE:
        correct_options = [
            _normalized(str(option.get("text", "")))
            for option in item.options
            if option.get("correct") is True
        ]
        correct = submitted in correct_options or submitted == expected
        return GradeResult(
            score=1.0 if correct else 0.0,
            correct=correct,
            correct_concepts=[item.answer] if correct else [],
            missing_concepts=[] if correct else [item.answer],
            mistaken_concepts=[]
            if correct or not submitted_answer.strip()
            else [submitted_answer],
            rationale=(
                "The response was compared with the single approved correct option."
            ),
            supporting_chunk_ids=source_ids,
        )

    expected_concepts = _concepts(item.answer)
    submitted_concepts = _concepts(submitted_answer)
    if not expected_concepts:
        score = 1.0 if submitted == expected and bool(expected) else 0.0
    else:
        score = len(expected_concepts & submitted_concepts) / len(expected_concepts)
    score = round(score, 3)
    correct = score >= (1.0 if item.item_type is StudyItemType.FLASHCARD else 0.8)
    present = sorted(expected_concepts & submitted_concepts)
    missing = sorted(expected_concepts - submitted_concepts)
    extra = sorted(submitted_concepts - expected_concepts)
    return GradeResult(
        score=score,
        correct=correct,
        correct_concepts=present,
        missing_concepts=missing,
        mistaken_concepts=extra,
        rationale=(
            "The response was scored by concept coverage against the approved answer; "
            "the cited passages below are the evidence for review."
        ),
        supporting_chunk_ids=source_ids,
    )

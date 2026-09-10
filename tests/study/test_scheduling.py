from datetime import UTC, datetime, timedelta

import pytest

from note_rag.persistence import (
    MemoryState,
    ReviewAttempt,
    ReviewRating,
    StudyItem,
    StudyItemType,
)
from note_rag.study import ForgettingCurveScheduler


def item() -> StudyItem:
    return StudyItem(
        topic_id=None,  # type: ignore[arg-type]
        item_type=StudyItemType.FLASHCARD,
        prompt="Prompt",
        answer="Answer",
        explanation="Explanation",
        options=[],
        difficulty=3,
        fingerprint="test",
    )


def attempt(at: datetime, *, correct: bool, rating: ReviewRating) -> ReviewAttempt:
    return ReviewAttempt(
        session_id=None,  # type: ignore[arg-type]
        study_item_id=None,  # type: ignore[arg-type]
        idempotency_key="attempt-key",
        submitted_answer="Answer",
        expected_answer="Answer",
        correct=correct,
        score=1.0 if correct else 0.0,
        rating=rating,
        confidence=3,
        response_time_ms=1000,
        hint_used=False,
        grading_details={},
        reviewed_at=at,
    )


def test_recall_declines_and_ratings_adjust_intervals() -> None:
    scheduler = ForgettingCurveScheduler()
    reviewed_at = datetime(2026, 9, 10, tzinfo=UTC)
    learning_item = item()

    good = scheduler.update(
        learning_item,
        attempt(reviewed_at, correct=True, rating=ReviewRating.GOOD),
        None,
    )
    later_recall = scheduler.predicted_recall(good, reviewed_at + timedelta(days=1))
    assert later_recall < 1

    failed = scheduler.update(
        learning_item,
        attempt(reviewed_at, correct=False, rating=ReviewRating.AGAIN),
        None,
    )
    assert failed.next_review_at < good.next_review_at


def test_scheduler_can_rebuild_state_from_attempts() -> None:
    scheduler = ForgettingCurveScheduler()
    start = datetime(2026, 9, 10, tzinfo=UTC)
    learning_item = item()
    attempts = [
        attempt(start, correct=False, rating=ReviewRating.AGAIN),
        attempt(start + timedelta(days=1), correct=True, rating=ReviewRating.EASY),
    ]

    rebuilt = scheduler.rebuild(learning_item, reversed(attempts))

    assert isinstance(rebuilt, MemoryState)
    assert rebuilt.successful_reviews == 1
    assert rebuilt.failed_reviews == 1
    assert rebuilt.last_review_at == start + timedelta(days=1)
    assert rebuilt.predicted_recall == pytest.approx(1)

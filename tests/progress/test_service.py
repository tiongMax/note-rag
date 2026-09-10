import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from note_rag.persistence import (
    ApprovalStatus,
    MemoryState,
    ReviewAttempt,
    ReviewRating,
    StudyItem,
    StudyItemType,
)
from note_rag.progress import ProgressService
from note_rag.study import SCHEDULER_VERSION, ForgettingCurveScheduler


def reviewed_item(*, difficulty: int, score: float, at: datetime) -> StudyItem:
    item_id = uuid.uuid4()
    item = StudyItem(
        id=item_id,
        topic_id=uuid.uuid4(),
        item_type=StudyItemType.FLASHCARD,
        prompt="Question",
        answer="Answer",
        explanation="Explanation",
        options=[],
        difficulty=difficulty,
        approval_status=ApprovalStatus.APPROVED,
        fingerprint=uuid.uuid4().hex,
    )
    item.review_attempts = [
        ReviewAttempt(
            session_id=uuid.uuid4(),
            study_item_id=item_id,
            idempotency_key=uuid.uuid4().hex,
            submitted_answer="Answer" if score else "Wrong",
            expected_answer="Answer",
            correct=score == 1,
            score=score,
            rating=ReviewRating.GOOD if score else ReviewRating.AGAIN,
            confidence=5,
            response_time_ms=1000,
            hint_used=False,
            grading_details={},
            reviewed_at=at,
        )
    ]
    item.memory_state = MemoryState(
        study_item_id=item_id,
        half_life_days=1,
        difficulty=difficulty,
        last_review_at=at,
        next_review_at=at + timedelta(days=1),
        predicted_recall=1,
        successful_reviews=int(score == 1),
        failed_reviews=int(score == 0),
        scheduler_version=SCHEDULER_VERSION,
    )
    return item


def test_progress_separates_coverage_mastery_and_retention() -> None:
    reviewed_at = datetime(2026, 9, 9, tzinfo=UTC)
    service = ProgressService(
        Session(),
        ForgettingCurveScheduler(),
        clock=reviewed_at + timedelta(days=1),
    )
    easy_success = reviewed_item(difficulty=1, score=1, at=reviewed_at)
    hard_failure = reviewed_item(difficulty=5, score=0, at=reviewed_at)

    summary = service.summarize_items([easy_success, hard_failure])

    assert summary.coverage == 1
    assert summary.mastery < 0.4  # The hard failure cannot be hidden by the easy card.
    assert summary.predicted_retention == 0.5
    assert summary.factors["weak_items"] == 1

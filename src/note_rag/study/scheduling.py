"""Versioned half-life scheduler backed by immutable review evidence."""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from note_rag.persistence import MemoryState, ReviewAttempt, ReviewRating, StudyItem

SCHEDULER_VERSION = "half-life-v1"


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    recall_threshold: float = 0.85
    minimum_interval_days: float = 10 / (24 * 60)
    maximum_interval_days: float = 365.0

    def __post_init__(self) -> None:
        if not 0 < self.recall_threshold < 1:
            raise ValueError("recall threshold must be between zero and one")
        if self.minimum_interval_days <= 0:
            raise ValueError("minimum interval must be positive")
        if self.maximum_interval_days < self.minimum_interval_days:
            raise ValueError("maximum interval must not be below the minimum")


class ForgettingCurveScheduler:
    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.config = config or SchedulerConfig()

    @staticmethod
    def predicted_recall(state: MemoryState, at: datetime) -> float:
        elapsed = max(0.0, (_aware(at) - _aware(state.last_review_at)).total_seconds())
        days = elapsed / 86_400
        return max(0.0, min(1.0, 0.5 ** (days / state.half_life_days)))

    def update(
        self,
        item: StudyItem,
        attempt: ReviewAttempt,
        state: MemoryState | None,
    ) -> MemoryState:
        effective_correct = (
            attempt.overridden_correct
            if attempt.overridden_correct is not None
            else attempt.correct
        )
        base = (
            state.half_life_days
            if state is not None
            else max(0.25, 1.5 - item.difficulty * 0.2)
        )
        multiplier = {
            ReviewRating.AGAIN: 0.35,
            ReviewRating.HARD: 1.2,
            ReviewRating.GOOD: 2.0,
            ReviewRating.EASY: 3.0,
        }[attempt.rating]
        if not effective_correct:
            multiplier = min(multiplier, 0.5)
        half_life = max(
            self.config.minimum_interval_days,
            min(self.config.maximum_interval_days, base * multiplier),
        )
        interval = half_life * math.log(self.config.recall_threshold, 0.5)
        interval = max(
            self.config.minimum_interval_days,
            min(self.config.maximum_interval_days, interval),
        )
        reviewed_at = _aware(attempt.reviewed_at)
        if state is None:
            state = MemoryState(
                study_item_id=item.id,
                half_life_days=half_life,
                difficulty=item.difficulty,
                last_review_at=reviewed_at,
                next_review_at=reviewed_at + timedelta(days=interval),
                predicted_recall=1.0,
                successful_reviews=0,
                failed_reviews=0,
                scheduler_version=SCHEDULER_VERSION,
            )
        state.half_life_days = half_life
        state.difficulty = item.difficulty
        state.last_review_at = reviewed_at
        state.next_review_at = reviewed_at + timedelta(days=interval)
        state.predicted_recall = 1.0
        state.scheduler_version = SCHEDULER_VERSION
        if effective_correct:
            state.successful_reviews += 1
        else:
            state.failed_reviews += 1
        return state

    def rebuild(
        self, item: StudyItem, attempts: Iterable[ReviewAttempt]
    ) -> MemoryState | None:
        state: MemoryState | None = None
        for attempt in sorted(
            attempts, key=lambda value: (_aware(value.reviewed_at), value.id)
        ):
            state = self.update(item, attempt, state)
        return state

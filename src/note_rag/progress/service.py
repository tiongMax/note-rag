"""Coverage, demonstrated mastery, and predicted retention calculations."""

import statistics
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from note_rag.persistence import (
    ApprovalStatus,
    Course,
    CourseRepository,
    MasteryScope,
    MasterySnapshot,
    MasterySnapshotRepository,
    ReviewAttempt,
    StudyItem,
    StudyItemRepository,
    Topic,
    TopicRepository,
)
from note_rag.study import ForgettingCurveScheduler


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _effective_score(attempt: ReviewAttempt) -> float:
    return (
        attempt.overridden_score
        if attempt.overridden_score is not None
        else attempt.score
    )


@dataclass(slots=True)
class ScoreSummary:
    coverage: float
    mastery: float
    predicted_retention: float
    encountered_items: int
    total_items: int
    factors: dict[str, float | int | str] = field(default_factory=dict)


class ProgressService:
    """Calculate explainable progress directly from immutable review evidence."""

    def __init__(
        self,
        session: Session,
        scheduler: ForgettingCurveScheduler,
        *,
        clock: datetime,
    ) -> None:
        self.session = session
        self.scheduler = scheduler
        self.clock = _aware(clock)

    def item_mastery(self, item: StudyItem) -> tuple[float, dict[str, float | int]]:
        attempts = sorted(
            item.review_attempts,
            key=lambda attempt: (_aware(attempt.reviewed_at), str(attempt.id)),
        )[-5:]
        if not attempts:
            return 0.0, {
                "recent_attempts": 0,
                "weighted_score": 0.0,
                "hint_penalty": 0.0,
                "confidence_calibration": 0.0,
            }
        weights = list(range(1, len(attempts) + 1))
        weighted_score = sum(
            _effective_score(attempt) * weight
            for attempt, weight in zip(attempts, weights, strict=True)
        ) / sum(weights)
        hint_rate = statistics.fmean(float(attempt.hint_used) for attempt in attempts)
        calibration_error = statistics.fmean(
            abs(attempt.confidence / 5 - _effective_score(attempt))
            for attempt in attempts
        )
        repetition_factor = 0.7 + 0.3 * min(1.0, len(attempts) / 3)
        mastery = weighted_score * (1 - 0.15 * hint_rate)
        mastery *= 1 - 0.15 * calibration_error
        mastery *= repetition_factor
        return max(0.0, min(1.0, mastery)), {
            "recent_attempts": len(attempts),
            "weighted_score": round(weighted_score, 4),
            "hint_penalty": round(0.15 * hint_rate, 4),
            "confidence_calibration": round(1 - 0.15 * calibration_error, 4),
        }

    def summarize_items(self, items: list[StudyItem]) -> ScoreSummary:
        approved = [
            item for item in items if item.approval_status is ApprovalStatus.APPROVED
        ]
        encountered = [item for item in approved if item.review_attempts]
        total = len(approved)
        weights = [1 + (item.difficulty - 1) * 0.25 for item in encountered]
        mastery_values = [self.item_mastery(item)[0] for item in encountered]
        mastery = (
            sum(
                value * weight
                for value, weight in zip(mastery_values, weights, strict=True)
            )
            / sum(weights)
            if weights
            else 0.0
        )
        retention_values = [
            self.scheduler.predicted_recall(item.memory_state, self.clock)
            for item in encountered
            if item.memory_state is not None
        ]
        retention = statistics.fmean(retention_values) if retention_values else 0.0
        weak_items = sum(value < 0.6 for value in mastery_values)
        return ScoreSummary(
            coverage=len(encountered) / total if total else 0.0,
            mastery=mastery,
            predicted_retention=retention,
            encountered_items=len(encountered),
            total_items=total,
            factors={
                "approved_items": total,
                "reviewed_items": len(encountered),
                "weak_items": weak_items,
                "difficulty_weighted": "true",
                "recent_attempt_window": 5,
            },
        )

    def topic(self, topic_id: uuid.UUID) -> tuple[Topic, ScoreSummary]:
        topic = TopicRepository(self.session).get(topic_id)
        if topic is None:
            raise LookupError("topic not found")
        return topic, self.summarize_items(topic.study_items)

    def course(
        self, course_id: uuid.UUID
    ) -> tuple[Course, ScoreSummary, list[tuple[Topic, ScoreSummary]]]:
        course = CourseRepository(self.session).get(course_id)
        if course is None:
            raise LookupError("course not found")
        items = StudyItemRepository(self.session).list_approved_for_course(course_id)
        summary = self.summarize_items(items)
        topics = [
            (topic, self.summarize_items(topic.study_items))
            for topic in sorted(
                course.topics, key=lambda value: (value.position, value.id)
            )
        ]
        return course, summary, topics

    def snapshot_course(self, course_id: uuid.UUID) -> MasterySnapshot:
        _course, summary, _topics = self.course(course_id)
        return MasterySnapshotRepository(self.session).add(
            MasterySnapshot(
                scope=MasteryScope.COURSE,
                course_id=course_id,
                coverage=summary.coverage,
                mastery=summary.mastery,
                predicted_retention=summary.predicted_retention,
                encountered_items=summary.encountered_items,
                total_items=summary.total_items,
                factors=summary.factors,
                captured_at=self.clock,
            )
        )

    def snapshot_topic(self, topic_id: uuid.UUID) -> MasterySnapshot:
        topic, summary = self.topic(topic_id)
        return MasterySnapshotRepository(self.session).add(
            MasterySnapshot(
                scope=MasteryScope.TOPIC,
                course_id=topic.course_id,
                topic_id=topic.id,
                coverage=summary.coverage,
                mastery=summary.mastery,
                predicted_retention=summary.predicted_retention,
                encountered_items=summary.encountered_items,
                total_items=summary.total_items,
                factors=summary.factors,
                captured_at=self.clock,
            )
        )

    def item_projection(
        self, item: StudyItem, *, days: int = 30
    ) -> list[dict[str, object]]:
        if item.memory_state is None:
            return []
        return [
            {
                "at": self.clock + timedelta(days=offset),
                "predicted_recall": self.scheduler.predicted_recall(
                    item.memory_state, self.clock + timedelta(days=offset)
                ),
            }
            for offset in range(days + 1)
        ]

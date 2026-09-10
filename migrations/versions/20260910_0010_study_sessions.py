"""Add study sessions, immutable attempts, and scheduler memory state."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260910_0010"
down_revision: str | None = "20260910_0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

session_mode = sa.Enum(
    "DAILY_REVIEW", "COURSE", "TOPIC", "SELECTED", name="study_session_mode"
)
session_status = sa.Enum("ACTIVE", "COMPLETED", name="study_session_status")
review_rating = sa.Enum("AGAIN", "HARD", "GOOD", "EASY", name="review_rating")


def timestamps() -> list[sa.Column]:
    return [
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    ]


def upgrade() -> None:
    op.create_table(
        "study_sessions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "course_id",
            sa.Uuid(),
            sa.ForeignKey("courses.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "topic_id", sa.Uuid(), sa.ForeignKey("topics.id", ondelete="SET NULL")
        ),
        sa.Column("mode", session_mode, nullable=False),
        sa.Column("status", session_status, server_default="ACTIVE", nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        *timestamps(),
    )
    op.create_index("ix_study_sessions_course_id", "study_sessions", ["course_id"])
    op.create_index("ix_study_sessions_status", "study_sessions", ["status"])
    op.create_table(
        "study_session_items",
        sa.Column(
            "session_id",
            sa.Uuid(),
            sa.ForeignKey("study_sessions.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "study_item_id",
            sa.Uuid(),
            sa.ForeignKey("study_items.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("reason", sa.String(255), nullable=False),
        sa.CheckConstraint("position >= 0", name="ck_study_session_items_position"),
        sa.UniqueConstraint("session_id", "position", name="uq_session_item_position"),
    )
    op.create_table(
        "review_attempts",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "session_id",
            sa.Uuid(),
            sa.ForeignKey("study_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "study_item_id",
            sa.Uuid(),
            sa.ForeignKey("study_items.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("idempotency_key", sa.String(128), nullable=False),
        sa.Column("submitted_answer", sa.Text(), nullable=False),
        sa.Column("expected_answer", sa.Text(), nullable=False),
        sa.Column("correct", sa.Boolean(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("rating", review_rating, nullable=False),
        sa.Column("confidence", sa.Integer(), nullable=False),
        sa.Column("response_time_ms", sa.Integer(), nullable=False),
        sa.Column("hint_used", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("grading_details", sa.JSON(), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("overridden_correct", sa.Boolean()),
        sa.Column("overridden_score", sa.Float()),
        sa.Column("override_reason", sa.Text()),
        sa.Column("overridden_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("score >= 0 AND score <= 1", name="ck_attempt_score"),
        sa.CheckConstraint(
            "confidence >= 1 AND confidence <= 5", name="ck_attempt_confidence"
        ),
        sa.CheckConstraint("response_time_ms >= 0", name="ck_attempt_response_time"),
        sa.UniqueConstraint(
            "session_id", "idempotency_key", name="uq_attempt_session_idempotency"
        ),
    )
    op.create_index(
        "ix_review_attempts_item_reviewed",
        "review_attempts",
        ["study_item_id", "reviewed_at"],
    )
    op.create_table(
        "memory_states",
        sa.Column(
            "study_item_id",
            sa.Uuid(),
            sa.ForeignKey("study_items.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("half_life_days", sa.Float(), nullable=False),
        sa.Column("difficulty", sa.Integer(), nullable=False),
        sa.Column("last_review_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("next_review_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("predicted_recall", sa.Float(), nullable=False),
        sa.Column(
            "successful_reviews", sa.Integer(), server_default="0", nullable=False
        ),
        sa.Column("failed_reviews", sa.Integer(), server_default="0", nullable=False),
        sa.Column("scheduler_version", sa.String(64), nullable=False),
        *timestamps(),
        sa.CheckConstraint("half_life_days > 0", name="ck_memory_half_life"),
        sa.CheckConstraint(
            "difficulty >= 1 AND difficulty <= 5", name="ck_memory_difficulty"
        ),
        sa.CheckConstraint(
            "predicted_recall >= 0 AND predicted_recall <= 1",
            name="ck_memory_predicted_recall",
        ),
        sa.CheckConstraint(
            "successful_reviews >= 0 AND failed_reviews >= 0",
            name="ck_memory_review_counts",
        ),
    )
    op.create_index("ix_memory_states_next_review", "memory_states", ["next_review_at"])


def downgrade() -> None:
    op.drop_table("memory_states")
    op.drop_table("review_attempts")
    op.drop_table("study_session_items")
    op.drop_table("study_sessions")
    review_rating.drop(op.get_bind(), checkfirst=True)
    session_status.drop(op.get_bind(), checkfirst=True)
    session_mode.drop(op.get_bind(), checkfirst=True)

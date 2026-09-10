"""Add generated learning objectives, study items, sources, and jobs."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260910_0009"
down_revision: str | None = "20260910_0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

study_item_type = sa.Enum(
    "FLASHCARD", "MULTIPLE_CHOICE", "SHORT_ANSWER", name="study_item_type"
)
approval_status = sa.Enum("DRAFT", "APPROVED", "ARCHIVED", name="approval_status")
generation_kind = sa.Enum(
    "CURRICULUM", "STUDY_ITEMS", "STUDY_ITEM", name="generation_kind"
)
generation_job_status = sa.Enum(
    "QUEUED", "RUNNING", "COMPLETED", "FAILED", name="generation_job_status"
)


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
        "learning_objectives",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "topic_id",
            sa.Uuid(),
            sa.ForeignKey("topics.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), server_default="", nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("generation_version", sa.String(128)),
        *timestamps(),
        sa.CheckConstraint("position >= 0", name="ck_learning_objectives_position"),
        sa.UniqueConstraint(
            "topic_id", "position", name="uq_learning_objectives_topic_position"
        ),
    )
    op.create_index(
        "ix_learning_objectives_topic_id", "learning_objectives", ["topic_id"]
    )
    op.create_table(
        "study_items",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "topic_id",
            sa.Uuid(),
            sa.ForeignKey("topics.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "objective_id",
            sa.Uuid(),
            sa.ForeignKey("learning_objectives.id", ondelete="SET NULL"),
        ),
        sa.Column("item_type", study_item_type, nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=False),
        sa.Column("options", sa.JSON(), nullable=False),
        sa.Column("difficulty", sa.Integer(), server_default="3", nullable=False),
        sa.Column(
            "approval_status", approval_status, server_default="DRAFT", nullable=False
        ),
        sa.Column("generation_version", sa.String(128)),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("archived_at", sa.DateTime(timezone=True)),
        *timestamps(),
        sa.CheckConstraint(
            "difficulty >= 1 AND difficulty <= 5", name="ck_study_items_difficulty"
        ),
    )
    op.create_index("ix_study_items_fingerprint", "study_items", ["fingerprint"])
    op.create_index(
        "ix_study_items_topic_status", "study_items", ["topic_id", "approval_status"]
    )
    op.create_table(
        "study_item_sources",
        sa.Column(
            "study_item_id",
            sa.Uuid(),
            sa.ForeignKey("study_items.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "chunk_id",
            sa.Uuid(),
            sa.ForeignKey("chunks.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        *timestamps(),
    )
    op.create_table(
        "generation_jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "course_id",
            sa.Uuid(),
            sa.ForeignKey("courses.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "topic_id", sa.Uuid(), sa.ForeignKey("topics.id", ondelete="CASCADE")
        ),
        sa.Column(
            "item_id", sa.Uuid(), sa.ForeignKey("study_items.id", ondelete="SET NULL")
        ),
        sa.Column("kind", generation_kind, nullable=False),
        sa.Column(
            "status", generation_job_status, server_default="QUEUED", nullable=False
        ),
        sa.Column("progress", sa.Integer(), server_default="0", nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("idempotency_key", sa.String(128), nullable=False),
        sa.Column("prompt_version", sa.String(128), nullable=False),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("error_message", sa.Text()),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        *timestamps(),
        sa.CheckConstraint(
            "progress >= 0 AND progress <= 100", name="ck_generation_jobs_progress"
        ),
        sa.CheckConstraint("attempts >= 0", name="ck_generation_jobs_attempts"),
        sa.UniqueConstraint(
            "idempotency_key", name="uq_generation_jobs_idempotency_key"
        ),
    )
    op.create_index("ix_generation_jobs_course_id", "generation_jobs", ["course_id"])
    op.create_index("ix_generation_jobs_status", "generation_jobs", ["status"])
    op.create_index(
        "ix_generation_jobs_status_created",
        "generation_jobs",
        ["status", "created_at"],
    )


def downgrade() -> None:
    op.drop_table("generation_jobs")
    op.drop_table("study_item_sources")
    op.drop_table("study_items")
    op.drop_table("learning_objectives")
    generation_job_status.drop(op.get_bind(), checkfirst=True)
    generation_kind.drop(op.get_bind(), checkfirst=True)
    approval_status.drop(op.get_bind(), checkfirst=True)
    study_item_type.drop(op.get_bind(), checkfirst=True)

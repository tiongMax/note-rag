"""Add immutable mastery and retention history snapshots."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260910_0011"
down_revision: str | None = "20260910_0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

mastery_scope = sa.Enum("COURSE", "TOPIC", name="mastery_scope")


def upgrade() -> None:
    op.create_table(
        "mastery_snapshots",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("scope", mastery_scope, nullable=False),
        sa.Column(
            "course_id",
            sa.Uuid(),
            sa.ForeignKey("courses.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "topic_id", sa.Uuid(), sa.ForeignKey("topics.id", ondelete="CASCADE")
        ),
        sa.Column("coverage", sa.Float(), nullable=False),
        sa.Column("mastery", sa.Float(), nullable=False),
        sa.Column("predicted_retention", sa.Float(), nullable=False),
        sa.Column("encountered_items", sa.Integer(), nullable=False),
        sa.Column("total_items", sa.Integer(), nullable=False),
        sa.Column("factors", sa.JSON(), nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "coverage >= 0 AND coverage <= 1", name="ck_snapshot_coverage"
        ),
        sa.CheckConstraint("mastery >= 0 AND mastery <= 1", name="ck_snapshot_mastery"),
        sa.CheckConstraint(
            "predicted_retention >= 0 AND predicted_retention <= 1",
            name="ck_snapshot_retention",
        ),
        sa.CheckConstraint(
            "encountered_items >= 0 AND total_items >= encountered_items",
            name="ck_snapshot_item_counts",
        ),
    )
    op.create_index(
        "ix_mastery_snapshots_course_captured",
        "mastery_snapshots",
        ["course_id", "captured_at"],
    )
    op.create_index(
        "ix_mastery_snapshots_topic_captured",
        "mastery_snapshots",
        ["topic_id", "captured_at"],
    )
    op.create_index(
        "ix_mastery_snapshots_captured_at", "mastery_snapshots", ["captured_at"]
    )


def downgrade() -> None:
    op.drop_table("mastery_snapshots")
    mastery_scope.drop(op.get_bind(), checkfirst=True)

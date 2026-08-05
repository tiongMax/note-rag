"""Add scheduling and lease fields for background ingestion jobs."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260805_0006"
down_revision: str | None = "20260805_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "ingestion_jobs",
        sa.Column("next_attempt_at", sa.DateTime(timezone=True)),
    )
    op.add_column(
        "ingestion_jobs",
        sa.Column("locked_at", sa.DateTime(timezone=True)),
    )
    op.add_column(
        "ingestion_jobs",
        sa.Column("worker_id", sa.String(length=64)),
    )
    op.create_index(
        "ix_ingestion_jobs_status_next_attempt",
        "ingestion_jobs",
        ["status", "next_attempt_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_ingestion_jobs_status_next_attempt",
        table_name="ingestion_jobs",
    )
    op.drop_column("ingestion_jobs", "worker_id")
    op.drop_column("ingestion_jobs", "locked_at")
    op.drop_column("ingestion_jobs", "next_attempt_at")

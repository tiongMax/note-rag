"""Add compact embedded memories for completed conversation turns."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260813_0009"
down_revision: str | None = "20260813_0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "conversation_memories",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("start_position", sa.Integer(), nullable=False),
        sa.Column("end_position", sa.Integer(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("embedding_model", sa.String(length=255), nullable=True),
        sa.Column("embedding_dimension", sa.Integer(), nullable=True),
        sa.Column("embedding", sa.JSON(none_as_null=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "embedding_dimension IS NULL OR embedding_dimension > 0",
            name="ck_conversation_memories_embedding_dimension",
        ),
        sa.CheckConstraint(
            "(embedding IS NULL AND embedding_model IS NULL AND "
            "embedding_dimension IS NULL) OR "
            "(embedding IS NOT NULL AND embedding_model IS NOT NULL AND "
            "embedding_dimension IS NOT NULL)",
            name="ck_conversation_memories_embedding_fields",
        ),
        sa.CheckConstraint(
            "end_position >= start_position",
            name="ck_conversation_memories_position_range",
        ),
        sa.CheckConstraint(
            "start_position >= 0",
            name="ck_conversation_memories_start_position",
        ),
        sa.CheckConstraint(
            "token_count >= 0",
            name="ck_conversation_memories_token_count",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "conversation_id",
            "start_position",
            "end_position",
            name="uq_conversation_memories_turn_range",
        ),
    )
    op.create_index(
        "ix_conversation_memories_conversation_position",
        "conversation_memories",
        ["conversation_id", "start_position"],
    )


def downgrade() -> None:
    op.drop_table("conversation_memories")

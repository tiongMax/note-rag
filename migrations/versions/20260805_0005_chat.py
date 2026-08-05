"""Add persisted conversations and chat messages."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260805_0005"
down_revision: str | None = "20260805_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

chat_role = sa.Enum("USER", "ASSISTANT", name="chat_role")


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
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
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_conversations_updated_at",
        "conversations",
        ["updated_at"],
    )
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("role", chat_role, nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("citations", sa.JSON(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("context_token_count", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=True),
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
            "context_token_count >= 0",
            name="ck_chat_messages_context_token_count",
        ),
        sa.CheckConstraint(
            "position >= 0",
            name="ck_chat_messages_position",
        ),
        sa.CheckConstraint(
            "token_count >= 0",
            name="ck_chat_messages_token_count",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "conversation_id",
            "position",
            name="uq_chat_messages_conversation_position",
        ),
    )
    op.create_index(
        "ix_chat_messages_conversation_created",
        "chat_messages",
        ["conversation_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_table("chat_messages")
    op.drop_table("conversations")
    chat_role.drop(op.get_bind(), checkfirst=True)

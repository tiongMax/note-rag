"""Add persistent query embedding and retrieval result caches."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260806_0007"
down_revision: str | None = "20260805_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "cache_state",
        sa.Column("key", sa.String(length=64), primary_key=True),
        sa.Column("value", sa.BigInteger(), nullable=False),
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
    )
    op.bulk_insert(
        sa.table(
            "cache_state",
            sa.column("key", sa.String()),
            sa.column("value", sa.BigInteger()),
        ),
        [{"key": "corpus_version", "value": 1}],
    )
    op.create_table(
        "query_embedding_cache",
        sa.Column("key_hash", sa.String(length=64), primary_key=True),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("normalized_query", sa.Text(), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("vector", sa.JSON(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
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
    )
    op.create_index(
        "ix_query_embedding_cache_expires",
        "query_embedding_cache",
        ["expires_at"],
    )
    op.create_table(
        "retrieval_result_cache",
        sa.Column("key_hash", sa.String(length=64), primary_key=True),
        sa.Column("corpus_version", sa.BigInteger(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
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
    )
    op.create_index(
        "ix_retrieval_result_cache_expires",
        "retrieval_result_cache",
        ["expires_at"],
    )
    op.create_index(
        "ix_retrieval_result_cache_corpus",
        "retrieval_result_cache",
        ["corpus_version"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_retrieval_result_cache_corpus",
        table_name="retrieval_result_cache",
    )
    op.drop_index(
        "ix_retrieval_result_cache_expires",
        table_name="retrieval_result_cache",
    )
    op.drop_table("retrieval_result_cache")
    op.drop_index(
        "ix_query_embedding_cache_expires",
        table_name="query_embedding_cache",
    )
    op.drop_table("query_embedding_cache")
    op.drop_table("cache_state")

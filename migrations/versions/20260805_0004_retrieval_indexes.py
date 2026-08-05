"""Add PostgreSQL indexes used by keyword and metadata retrieval."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260805_0004"
down_revision: str | None = "20260805_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "chunks",
        "source_metadata",
        existing_type=sa.JSON(),
        type_=postgresql.JSONB(astext_type=sa.Text()),
        postgresql_using="source_metadata::jsonb",
        existing_nullable=False,
    )
    op.create_index(
        "ix_chunks_source_metadata_gin",
        "chunks",
        ["source_metadata"],
        postgresql_using="gin",
    )
    op.execute(
        "CREATE INDEX ix_chunks_text_fts ON chunks "
        "USING gin (to_tsvector('english'::regconfig, text))"
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_text_fts", table_name="chunks")
    op.drop_index("ix_chunks_source_metadata_gin", table_name="chunks")
    op.alter_column(
        "chunks",
        "source_metadata",
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        type_=sa.JSON(),
        postgresql_using="source_metadata::json",
        existing_nullable=False,
    )

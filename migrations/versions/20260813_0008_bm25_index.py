"""Add persisted lexical statistics for PostgreSQL BM25 retrieval."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260813_0008"
down_revision: str | None = "20260806_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column(
            "lexical_token_count",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
    )
    op.create_table(
        "chunk_lexical_terms",
        sa.Column("chunk_id", sa.Uuid(), nullable=False),
        sa.Column("term", sa.String(length=128), nullable=False),
        sa.Column("term_frequency", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "term_frequency > 0",
            name="ck_chunk_lexical_terms_frequency",
        ),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("chunk_id", "term"),
    )
    op.create_index(
        "ix_chunk_lexical_terms_term",
        "chunk_lexical_terms",
        ["term"],
    )
    op.execute(
        """
        UPDATE chunks AS chunk
        SET lexical_token_count = stats.term_count
        FROM (
            SELECT chunks.id, count(term)::integer AS term_count
            FROM chunks
            CROSS JOIN LATERAL regexp_split_to_table(
                lower(chunks.text), '[^[:alnum:]_]+'
            ) AS term
            WHERE term <> ''
            GROUP BY chunks.id
        ) AS stats
        WHERE chunk.id = stats.id
        """
    )
    op.execute(
        """
        INSERT INTO chunk_lexical_terms (chunk_id, term, term_frequency)
        SELECT chunks.id, left(term, 128), count(*)::integer
        FROM chunks
        CROSS JOIN LATERAL regexp_split_to_table(
            lower(chunks.text), '[^[:alnum:]_]+'
        ) AS term
        WHERE term <> ''
        GROUP BY chunks.id, left(term, 128)
        """
    )
    op.alter_column("chunks", "lexical_token_count", server_default=None)


def downgrade() -> None:
    op.drop_index(
        "ix_chunk_lexical_terms_term",
        table_name="chunk_lexical_terms",
    )
    op.drop_table("chunk_lexical_terms")
    op.drop_column("chunks", "lexical_token_count")

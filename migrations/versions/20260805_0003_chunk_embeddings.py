"""Add pgvector chunk embeddings and indexing state."""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

revision: str = "20260805_0003"
down_revision: str | None = "20260805_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

indexing_status = sa.Enum(
    "PENDING", "INDEXING", "INDEXED", "FAILED", name="indexing_status"
)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    indexing_status.create(op.get_bind(), checkfirst=True)
    op.add_column(
        "documents",
        sa.Column(
            "indexing_status",
            indexing_status,
            server_default="PENDING",
            nullable=False,
        ),
    )
    op.add_column("documents", sa.Column("embedding_model", sa.String(255)))
    op.add_column(
        "documents", sa.Column("indexed_at", sa.DateTime(timezone=True))
    )
    op.add_column("documents", sa.Column("indexing_error", sa.Text()))
    op.create_index(
        "ix_documents_indexing_status", "documents", ["indexing_status"]
    )
    op.add_column(
        "chunks",
        sa.Column("embedding", pgvector.sqlalchemy.Vector(dim=768)),
    )
    op.add_column("chunks", sa.Column("embedding_model", sa.String(255)))
    op.add_column("chunks", sa.Column("embedded_at", sa.DateTime(timezone=True)))
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_embedding_hnsw", table_name="chunks")
    op.drop_column("chunks", "embedded_at")
    op.drop_column("chunks", "embedding_model")
    op.drop_column("chunks", "embedding")
    op.drop_index("ix_documents_indexing_status", table_name="documents")
    op.drop_column("documents", "indexing_error")
    op.drop_column("documents", "indexed_at")
    op.drop_column("documents", "embedding_model")
    op.drop_column("documents", "indexing_status")
    indexing_status.drop(op.get_bind(), checkfirst=True)

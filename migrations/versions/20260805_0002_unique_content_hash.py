"""Make document content hashes unique for duplicate detection."""

from collections.abc import Sequence

from alembic import op

revision: str = "20260805_0002"
down_revision: str | None = "20260805_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "uq_documents_content_hash",
        "documents",
        ["content_hash"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_documents_content_hash", table_name="documents")

"""Add per-chunk OCR/extraction warnings."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002_chunk_extraction_warnings"
down_revision = "0001_initial_storage"
branch_labels = None
depends_on = None

CONSTRAINT = "ck_chunks_extraction_warnings_string_array"


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column(
            "extraction_warnings",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
    op.create_check_constraint(
        CONSTRAINT,
        "chunks",
        "jsonb_typeof(extraction_warnings) = 'array' AND "
        "NOT jsonb_path_exists(extraction_warnings, '$[*] ? (@.type() != \"string\")')",
    )


def downgrade() -> None:
    op.drop_constraint(CONSTRAINT, "chunks", type_="check")
    op.drop_column("chunks", "extraction_warnings")

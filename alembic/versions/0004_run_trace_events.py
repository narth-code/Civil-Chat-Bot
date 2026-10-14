"""Persist structured agent execution traces.

Revision ID: 0004_run_trace_events
Revises: 0003_agent_answers
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0004_run_trace_events"
down_revision = "0003_agent_answers"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "run_trace_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("node", sa.Text(), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("data", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.CheckConstraint("seq >= 0", name="ck_run_trace_events_seq_nonnegative"),
        sa.CheckConstraint("duration_ms >= 0", name="ck_run_trace_events_duration_nonnegative"),
        sa.CheckConstraint(
            "kind IN ('node', 'llm_call', 'retrieval', 'web_search', 'decision')",
            name="ck_run_trace_events_kind",
        ),
        sa.ForeignKeyConstraint(["run_id"], ["routing_runs.id"], name="fk_run_trace_events_run", ondelete="RESTRICT"),
        sa.UniqueConstraint("run_id", "seq", name="uq_run_trace_events_run_seq"),
    )
    op.create_index("ix_run_trace_events_run_seq", "run_trace_events", ["run_id", "seq"])


def downgrade() -> None:
    op.drop_index("ix_run_trace_events_run_seq", table_name="run_trace_events")
    op.drop_table("run_trace_events")

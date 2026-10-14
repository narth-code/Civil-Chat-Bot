"""Persist agent outcomes and linked assistant answers without rewriting historical runs.

Revision ID: 0003_agent_answers
Revises: 0002_chunk_extraction_warnings
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0003_agent_answers"
down_revision = "0002_chunk_extraction_warnings"
branch_labels = None
depends_on = None

OLD_ORIGIN = "origin IN ('user', 'router_status', 'legacy_demo')"
NEW_ORIGIN = "origin IN ('user', 'router_status', 'assistant_answer', 'legacy_demo')"
OLD_ROLE = "(role = 'user' AND origin = 'user') OR (role = 'assistant' AND origin IN ('router_status', 'legacy_demo'))"
NEW_ROLE = "(role = 'user' AND origin = 'user') OR (role = 'assistant' AND origin IN ('router_status', 'assistant_answer', 'legacy_demo'))"
OLD_LINK = "(origin = 'router_status') = (routing_run_id IS NOT NULL)"
NEW_LINK = "(origin IN ('router_status', 'assistant_answer')) = (routing_run_id IS NOT NULL)"


def _constraints(origin: str, role: str, link: str) -> None:
    op.create_check_constraint("ck_messages_origin", "messages", origin)
    op.create_check_constraint("ck_messages_role_origin", "messages", role)
    op.create_check_constraint("ck_messages_status_run_link", "messages", link)


def upgrade() -> None:
    op.add_column("routing_runs", sa.Column("outcome", sa.String(32), nullable=True))
    op.add_column(
        "routing_runs",
        sa.Column("citations", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
    )
    op.add_column(
        "routing_runs", sa.Column("trace", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb"))
    )
    op.add_column("routing_runs", sa.Column("review", postgresql.JSONB(), nullable=True))
    op.add_column("routing_runs", sa.Column("city", sa.Text(), nullable=True))
    op.create_check_constraint(
        "ck_routing_runs_outcome",
        "routing_runs",
        "outcome IS NULL OR outcome IN ('kb_answer', 'web_answer', 'clarification', 'blocked', 'insufficient', 'direct', 'error')",
    )
    op.create_check_constraint(
        "ck_routing_runs_arrays", "routing_runs", "jsonb_typeof(citations) = 'array' AND jsonb_typeof(trace) = 'array'"
    )
    for name in ("ck_messages_origin", "ck_messages_role_origin", "ck_messages_status_run_link"):
        op.drop_constraint(name, "messages", type_="check")
    _constraints(NEW_ORIGIN, NEW_ROLE, NEW_LINK)
    op.drop_index("uq_messages_router_status_per_run", table_name="messages")
    op.create_index(
        "uq_messages_router_status_per_run",
        "messages",
        ["routing_run_id"],
        unique=True,
        postgresql_where=sa.text("origin IN ('router_status', 'assistant_answer')"),
    )


def downgrade() -> None:
    # A downgrade cannot represent new assistant answers; refuse data loss explicitly.
    connection = op.get_bind()
    if connection.execute(sa.text("SELECT EXISTS (SELECT 1 FROM messages WHERE origin = 'assistant_answer')")).scalar():
        raise RuntimeError("Cannot downgrade while agent assistant answers exist")
    op.drop_index("uq_messages_router_status_per_run", table_name="messages")
    op.create_index(
        "uq_messages_router_status_per_run",
        "messages",
        ["routing_run_id"],
        unique=True,
        postgresql_where=sa.text("origin = 'router_status'"),
    )
    for name in ("ck_messages_origin", "ck_messages_role_origin", "ck_messages_status_run_link"):
        op.drop_constraint(name, "messages", type_="check")
    _constraints(OLD_ORIGIN, OLD_ROLE, OLD_LINK)
    op.drop_constraint("ck_routing_runs_arrays", "routing_runs", type_="check")
    op.drop_constraint("ck_routing_runs_outcome", "routing_runs", type_="check")
    for name in ("city", "review", "trace", "citations", "outcome"):
        op.drop_column("routing_runs", name)

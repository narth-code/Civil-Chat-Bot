"""Create the immutable source inventory and routing-audit schema."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001_initial_storage"
down_revision = None
branch_labels = None
depends_on = None


class Vector(sa.types.UserDefinedType):
    """Unconstrained pgvector `vector` column type, declared locally so the migration does not import app code."""

    cache_ok = True

    def get_col_spec(self, **kw: object) -> str:
        return "vector"


NOW = sa.text("now()")
EMPTY_ARRAY = sa.text("'[]'::jsonb")


def _string_array(column: str) -> str:
    return f"jsonb_typeof({column}) = 'array' AND NOT jsonb_path_exists({column}, '$[*] ? (@.type() != \"string\")')"


def _uuid_pk() -> sa.Column:
    return sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True)


def upgrade() -> None:
    # Only the initial schema owner enables pgvector; application startup never does.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "routing_policies",
        sa.Column("version", sa.String(100), primary_key=True),
        sa.Column("system_prompt", sa.Text(), nullable=False),
        sa.Column("response_schema", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
    )

    op.create_table(
        "conversations",
        _uuid_pk(),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
    )

    # documents <-> document_versions reference each other; the documents FK is added afterwards.
    op.create_table(
        "documents",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("label", sa.String(200), nullable=False),
        sa.Column("current_source_version_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_table(
        "document_versions",
        _uuid_pk(),
        sa.Column("document_id", sa.String(64), sa.ForeignKey("documents.id"), nullable=False),
        sa.Column("source_title", sa.String(500), nullable=False),
        sa.Column("jurisdiction", sa.String(300), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("snapshot_path", sa.String(600), nullable=False),
        sa.Column("edition", sa.String(300), nullable=True),
        sa.Column("effective_date", sa.Date(), nullable=True),
        sa.Column("applicability_note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.UniqueConstraint("document_id", "sha256", name="uq_document_versions_document_sha256"),
        sa.UniqueConstraint("id", "document_id", name="uq_document_versions_id_document"),
    )
    op.create_foreign_key(
        "fk_documents_current_source_version_same_document",
        "documents",
        "document_versions",
        ["current_source_version_id", "id"],
        ["id", "document_id"],
    )

    op.create_table(
        "provisions",
        _uuid_pk(),
        sa.Column(
            "source_version_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("document_versions.id"), nullable=False
        ),
        sa.Column("parent_provision_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("full_identifier", sa.String(300), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("sibling_order", sa.Integer(), nullable=False),
        sa.UniqueConstraint("id", "source_version_id", name="uq_provisions_id_source_version"),
        sa.UniqueConstraint("source_version_id", "full_identifier", name="uq_provisions_source_version_identifier"),
        sa.ForeignKeyConstraint(
            ["parent_provision_id", "source_version_id"],
            ["provisions.id", "provisions.source_version_id"],
            name="fk_provisions_parent_same_source_version",
        ),
        sa.CheckConstraint(
            "parent_provision_id IS NULL OR parent_provision_id <> id", name="ck_provisions_parent_not_self"
        ),
        sa.CheckConstraint("sibling_order >= 0", name="ck_provisions_sibling_order_nonnegative"),
    )

    op.create_table(
        "chunks",
        _uuid_pk(),
        sa.Column(
            "source_version_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("document_versions.id"), nullable=False
        ),
        sa.Column("provision_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("physical_page_start", sa.Integer(), nullable=True),
        sa.Column("physical_page_end", sa.Integer(), nullable=True),
        sa.Column("printed_page_label", sa.String(100), nullable=True),
        sa.Column("table_markdown", sa.Text(), nullable=True),
        sa.Column("extraction_method", sa.String(100), nullable=True),
        sa.Column("extraction_confidence", sa.Float(), nullable=True),
        sa.Column("quality_flags", postgresql.JSONB(astext_type=sa.Text()), server_default=EMPTY_ARRAY, nullable=False),
        sa.UniqueConstraint("provision_id", "ordinal", name="uq_chunks_provision_ordinal"),
        sa.ForeignKeyConstraint(
            ["provision_id", "source_version_id"],
            ["provisions.id", "provisions.source_version_id"],
            name="fk_chunks_provision_same_source_version",
        ),
        sa.CheckConstraint("ordinal >= 0", name="ck_chunks_ordinal_nonnegative"),
        sa.CheckConstraint(
            "(physical_page_start IS NULL AND physical_page_end IS NULL) OR "
            "(physical_page_start IS NOT NULL AND physical_page_end IS NOT NULL AND physical_page_start > 0 "
            "AND physical_page_end > 0 AND physical_page_start <= physical_page_end)",
            name="ck_chunks_physical_pages",
        ),
        sa.CheckConstraint(
            "extraction_confidence IS NULL OR (extraction_confidence >= 0 AND extraction_confidence <= 1)",
            name="ck_chunks_extraction_confidence",
        ),
        sa.CheckConstraint(_string_array("quality_flags"), name="ck_chunks_quality_flags_string_array"),
    )
    op.create_index(
        "ix_chunks_raw_text_tsv",
        "chunks",
        [sa.text("to_tsvector('english', raw_text)")],
        postgresql_using="gin",
    )

    # The vector column is deliberately unconstrained until an embedding model/dimension is chosen.
    op.create_table(
        "chunk_embeddings",
        _uuid_pk(),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("chunks.id"), nullable=False),
        sa.Column("model", sa.String(300), nullable=False),
        sa.Column("revision", sa.String(300), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("embedding_input_sha256", sa.String(64), nullable=False),
        sa.Column("embedding", Vector(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.UniqueConstraint(
            "chunk_id", "model", "revision", "embedding_input_sha256", name="uq_chunk_embeddings_input"
        ),
        sa.CheckConstraint("dimensions > 0", name="ck_chunk_embeddings_dimensions_positive"),
        sa.CheckConstraint("dimensions = vector_dims(embedding)", name="ck_chunk_embeddings_vector_dimensions"),
    )

    # messages <-> routing_runs reference each other; the messages.routing_run_id FK is added afterwards.
    op.create_table(
        "messages",
        _uuid_pk(),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("conversations.id"), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.Column("origin", sa.String(32), nullable=False),
        sa.Column(
            "selected_manual_ids", postgresql.JSONB(astext_type=sa.Text()), server_default=EMPTY_ARRAY, nullable=False
        ),
        sa.Column("routing_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.UniqueConstraint("conversation_id", "sequence", name="uq_messages_conversation_sequence"),
        sa.UniqueConstraint("id", "conversation_id", name="uq_messages_id_conversation"),
        sa.CheckConstraint("sequence >= 0", name="ck_messages_sequence_nonnegative"),
        sa.CheckConstraint("role IN ('user', 'assistant')", name="ck_messages_role"),
        sa.CheckConstraint("origin IN ('user', 'router_status', 'legacy_demo')", name="ck_messages_origin"),
        sa.CheckConstraint(
            "(role = 'user' AND origin = 'user') OR (role = 'assistant' AND origin IN ('router_status', 'legacy_demo'))",
            name="ck_messages_role_origin",
        ),
        sa.CheckConstraint(
            "(origin = 'router_status') = (routing_run_id IS NOT NULL)", name="ck_messages_status_run_link"
        ),
        sa.CheckConstraint(_string_array("selected_manual_ids"), name="ck_messages_selected_manual_ids_string_array"),
    )
    op.create_index("ix_messages_conversation_sequence", "messages", ["conversation_id", "sequence"])
    op.create_index(
        "uq_messages_router_status_per_run",
        "messages",
        ["routing_run_id"],
        unique=True,
        postgresql_where=sa.text("origin = 'router_status'"),
    )

    op.create_table(
        "routing_runs",
        _uuid_pk(),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("conversations.id"), nullable=False),
        sa.Column("user_message_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("policy_version", sa.String(100), sa.ForeignKey("routing_policies.version"), nullable=False),
        sa.Column("model", sa.String(300), nullable=False),
        sa.Column("model_digest", sa.String(300), nullable=True),
        sa.Column("model_options", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("route", sa.String(32), nullable=True),
        sa.Column("reason_code", sa.String(64), nullable=True),
        sa.Column("rationale", sa.String(400), nullable=True),
        sa.Column("error_code", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("elapsed_ms", sa.Integer(), nullable=True),
        sa.Column("routing_ms", sa.Integer(), nullable=True),
        sa.Column("model_calls", sa.Integer(), nullable=False),
        sa.UniqueConstraint("id", "conversation_id", name="uq_routing_runs_id_conversation"),
        sa.ForeignKeyConstraint(
            ["user_message_id", "conversation_id"],
            ["messages.id", "messages.conversation_id"],
            name="fk_routing_runs_user_message_same_conversation",
        ),
        sa.CheckConstraint("status IN ('pending', 'classified', 'failed')", name="ck_routing_runs_status"),
        sa.CheckConstraint(
            "route IS NULL OR "
            "(route = 'retrieval' AND reason_code = 'code_lookup') OR "
            "(route = 'conversation' AND reason_code IN ('greeting', 'clarification', 'workspace_help')) OR "
            "(route = 'invalid' AND reason_code IN ('unsafe_request', 'prompt_injection', 'off_topic'))",
            name="ck_routing_runs_route_reason",
        ),
        sa.CheckConstraint(
            "(status = 'pending' AND route IS NULL AND reason_code IS NULL AND rationale IS NULL "
            "AND error_code IS NULL AND completed_at IS NULL) OR "
            "(status = 'classified' AND route IS NOT NULL AND reason_code IS NOT NULL AND rationale IS NOT NULL "
            "AND error_code IS NULL AND completed_at IS NOT NULL) OR "
            "(status = 'failed' AND route IS NULL AND reason_code IS NULL AND rationale IS NULL "
            "AND error_code IS NOT NULL AND completed_at IS NOT NULL)",
            name="ck_routing_runs_status_fields",
        ),
        sa.CheckConstraint(
            "rationale IS NULL OR char_length(rationale) <= 400", name="ck_routing_runs_rationale_length"
        ),
        sa.CheckConstraint("elapsed_ms IS NULL OR elapsed_ms >= 0", name="ck_routing_runs_elapsed_nonnegative"),
        sa.CheckConstraint("routing_ms IS NULL OR routing_ms >= 0", name="ck_routing_runs_routing_nonnegative"),
        sa.CheckConstraint("model_calls IN (0, 1)", name="ck_routing_runs_model_calls"),
    )
    op.create_index("ix_routing_runs_created_status", "routing_runs", ["created_at", "status"])
    op.create_index(
        "uq_routing_runs_pending_conversation",
        "routing_runs",
        ["conversation_id"],
        unique=True,
        postgresql_where=sa.text("status = 'pending'"),
    )
    op.create_foreign_key(
        "fk_messages_routing_run_same_conversation",
        "messages",
        "routing_runs",
        ["routing_run_id", "conversation_id"],
        ["id", "conversation_id"],
    )

    op.create_table(
        "routing_run_context_messages",
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("message_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.PrimaryKeyConstraint("run_id", "ordinal", name="uq_routing_run_context_ordinal"),
        sa.UniqueConstraint("run_id", "message_id", name="uq_routing_run_context_message"),
        sa.ForeignKeyConstraint(
            ["run_id", "conversation_id"],
            ["routing_runs.id", "routing_runs.conversation_id"],
            name="fk_routing_context_run_same_conversation",
        ),
        sa.ForeignKeyConstraint(
            ["message_id", "conversation_id"],
            ["messages.id", "messages.conversation_id"],
            name="fk_routing_context_message_same_conversation",
        ),
        sa.CheckConstraint("ordinal >= 0", name="ck_routing_context_ordinal_nonnegative"),
    )

    op.create_table(
        "routing_run_sources",
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.String(64), nullable=False),
        sa.Column("source_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.PrimaryKeyConstraint("run_id", "ordinal", name="uq_routing_run_sources_ordinal"),
        sa.UniqueConstraint("run_id", "document_id", name="uq_routing_run_sources_document"),
        sa.ForeignKeyConstraint(["run_id"], ["routing_runs.id"], name="fk_routing_sources_run"),
        sa.ForeignKeyConstraint(
            ["source_version_id", "document_id"],
            ["document_versions.id", "document_versions.document_id"],
            name="fk_routing_sources_version_same_document",
        ),
        sa.CheckConstraint("ordinal >= 0", name="ck_routing_sources_ordinal_nonnegative"),
    )

    op.create_table(
        "routing_evaluations",
        _uuid_pk(),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("routing_runs.id"), nullable=False),
        sa.Column("expected_route", sa.String(32), nullable=False),
        sa.Column("comment", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=NOW, nullable=False),
        sa.CheckConstraint("char_length(comment) <= 5000", name="ck_routing_evaluations_comment_length"),
    )
    op.create_index(
        "ix_routing_evaluations_run_latest",
        "routing_evaluations",
        ["run_id", sa.text("created_at DESC"), sa.text("id DESC")],
    )


def downgrade() -> None:
    op.drop_table("routing_evaluations")
    op.drop_table("routing_run_sources")
    op.drop_table("routing_run_context_messages")
    op.drop_constraint("fk_messages_routing_run_same_conversation", "messages", type_="foreignkey")
    op.drop_table("routing_runs")
    op.drop_table("messages")
    op.drop_table("chunk_embeddings")
    op.drop_table("chunks")
    op.drop_table("provisions")
    op.drop_constraint("fk_documents_current_source_version_same_document", "documents", type_="foreignkey")
    op.drop_table("document_versions")
    op.drop_table("documents")
    op.drop_table("conversations")
    op.drop_table("routing_policies")
    # The vector extension is left installed: it is database-wide and may be used outside this schema.

"""SQLAlchemy mappings for immutable source inventory and routing audit records."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class MessageOrigin(str, Enum):
    USER = "user"
    ROUTER_STATUS = "router_status"
    ASSISTANT_ANSWER = "assistant_answer"
    LEGACY_DEMO = "legacy_demo"


class RoutingRunStatus(str, Enum):
    PENDING = "pending"
    CLASSIFIED = "classified"
    FAILED = "failed"


UTC_NOW = text("now()")
JSON_STRING_ARRAY = (
    "jsonb_typeof({column}) = 'array' AND NOT jsonb_path_exists({column}, '$[*] ? (@.type() != \"string\")')"
)


# --- Source inventory: registered manuals, their immutable versions, and retrievable text ---


class Document(Base):
    """A manual; its current version pointer must reference one of its own versions."""

    __tablename__ = "documents"
    __table_args__ = (
        ForeignKeyConstraint(
            ["current_source_version_id", "id"],
            ["document_versions.id", "document_versions.document_id"],
            name="fk_documents_current_source_version_same_document",
            use_alter=True,
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    label: Mapped[str] = mapped_column(String(200), nullable=False)
    current_source_version_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)

    versions: Mapped[list[DocumentVersion]] = relationship(
        back_populates="document", foreign_keys="DocumentVersion.document_id"
    )
    current_source_version: Mapped[DocumentVersion | None] = relationship(
        foreign_keys=[current_source_version_id], post_update=True
    )


class DocumentVersion(Base):
    """One immutable PDF snapshot of a document, unique per content hash."""

    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("id", "document_id", name="uq_document_versions_id_document"),
        UniqueConstraint("document_id", "sha256", name="uq_document_versions_document_sha256"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), nullable=False)
    source_title: Mapped[str] = mapped_column(String(500), nullable=False)
    jurisdiction: Mapped[str] = mapped_column(String(300), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    snapshot_path: Mapped[str] = mapped_column(String(600), nullable=False)
    edition: Mapped[str | None] = mapped_column(String(300), nullable=True)
    effective_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    applicability_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=UTC_NOW)

    document: Mapped[Document] = relationship(back_populates="versions", foreign_keys=[document_id])


class Provision(Base):
    """A code provision in a version's hierarchy; parent must be a different provision of the same version."""

    __tablename__ = "provisions"
    __table_args__ = (
        UniqueConstraint("id", "source_version_id", name="uq_provisions_id_source_version"),
        UniqueConstraint("source_version_id", "full_identifier", name="uq_provisions_source_version_identifier"),
        ForeignKeyConstraint(
            ["parent_provision_id", "source_version_id"],
            ["provisions.id", "provisions.source_version_id"],
            name="fk_provisions_parent_same_source_version",
        ),
        CheckConstraint(
            "parent_provision_id IS NULL OR parent_provision_id <> id", name="ck_provisions_parent_not_self"
        ),
        CheckConstraint("sibling_order >= 0", name="ck_provisions_sibling_order_nonnegative"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    source_version_id: Mapped[UUID] = mapped_column(ForeignKey("document_versions.id"), nullable=False)
    parent_provision_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    full_identifier: Mapped[str] = mapped_column(String(300), nullable=False)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    sibling_order: Mapped[int] = mapped_column(Integer, nullable=False)


class Chunk(Base):
    """Exact source text of a provision; its version must match the provision's, with valid page bounds and confidence."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("provision_id", "ordinal", name="uq_chunks_provision_ordinal"),
        ForeignKeyConstraint(
            ["provision_id", "source_version_id"],
            ["provisions.id", "provisions.source_version_id"],
            name="fk_chunks_provision_same_source_version",
        ),
        CheckConstraint("ordinal >= 0", name="ck_chunks_ordinal_nonnegative"),
        CheckConstraint(
            "(physical_page_start IS NULL AND physical_page_end IS NULL) OR "
            "(physical_page_start IS NOT NULL AND physical_page_end IS NOT NULL AND physical_page_start > 0 AND physical_page_end > 0 AND physical_page_start <= physical_page_end)",
            name="ck_chunks_physical_pages",
        ),
        CheckConstraint(
            "extraction_confidence IS NULL OR (extraction_confidence >= 0 AND extraction_confidence <= 1)",
            name="ck_chunks_extraction_confidence",
        ),
        CheckConstraint(JSON_STRING_ARRAY.format(column="quality_flags"), name="ck_chunks_quality_flags_string_array"),
        CheckConstraint(
            JSON_STRING_ARRAY.format(column="extraction_warnings"),
            name="ck_chunks_extraction_warnings_string_array",
        ),
        Index("ix_chunks_raw_text_tsv", text("to_tsvector('english', raw_text)"), postgresql_using="gin"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    source_version_id: Mapped[UUID] = mapped_column(ForeignKey("document_versions.id"), nullable=False)
    provision_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    physical_page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    physical_page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    printed_page_label: Mapped[str | None] = mapped_column(String(100), nullable=True)
    table_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    extraction_method: Mapped[str | None] = mapped_column(String(100), nullable=True)
    extraction_confidence: Mapped[float | None] = mapped_column(nullable=True)
    quality_flags: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    extraction_warnings: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )


class ChunkEmbedding(Base):
    """A model-specific embedding of a chunk whose declared dimensions match the stored vector."""

    __tablename__ = "chunk_embeddings"
    __table_args__ = (
        UniqueConstraint("chunk_id", "model", "revision", "embedding_input_sha256", name="uq_chunk_embeddings_input"),
        CheckConstraint("dimensions > 0", name="ck_chunk_embeddings_dimensions_positive"),
        CheckConstraint("dimensions = vector_dims(embedding)", name="ck_chunk_embeddings_vector_dimensions"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    chunk_id: Mapped[UUID] = mapped_column(ForeignKey("chunks.id"), nullable=False)
    model: Mapped[str] = mapped_column(String(300), nullable=False)
    revision: Mapped[str] = mapped_column(String(300), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_input_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding: Mapped[Any] = mapped_column(Vector(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=UTC_NOW)


# --- Conversations: saved chat history ---


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    title: Mapped[str] = mapped_column(String(500), nullable=False, default="New conversation")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=UTC_NOW)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=UTC_NOW, onupdate=UTC_NOW
    )

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation", foreign_keys="Message.conversation_id"
    )
    routing_runs: Mapped[list[RoutingRun]] = relationship(
        back_populates="conversation", foreign_keys="RoutingRun.conversation_id"
    )


class Message(Base):
    """A conversation message; sequence is unique per conversation and role/origin pairs are constrained."""

    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("id", "conversation_id", name="uq_messages_id_conversation"),
        UniqueConstraint("conversation_id", "sequence", name="uq_messages_conversation_sequence"),
        ForeignKeyConstraint(
            ["routing_run_id", "conversation_id"],
            ["routing_runs.id", "routing_runs.conversation_id"],
            name="fk_messages_routing_run_same_conversation",
            use_alter=True,
        ),
        CheckConstraint("sequence >= 0", name="ck_messages_sequence_nonnegative"),
        CheckConstraint("role IN ('user', 'assistant')", name="ck_messages_role"),
        CheckConstraint(
            "origin IN ('user', 'router_status', 'assistant_answer', 'legacy_demo')", name="ck_messages_origin"
        ),
        CheckConstraint(
            "(role = 'user' AND origin = 'user') OR (role = 'assistant' AND origin IN ('router_status', 'assistant_answer', 'legacy_demo'))",
            name="ck_messages_role_origin",
        ),
        CheckConstraint(
            "(origin IN ('router_status', 'assistant_answer')) = (routing_run_id IS NOT NULL)",
            name="ck_messages_status_run_link",
        ),
        CheckConstraint(
            JSON_STRING_ARRAY.format(column="selected_manual_ids"), name="ck_messages_selected_manual_ids_string_array"
        ),
        Index("ix_messages_conversation_sequence", "conversation_id", "sequence"),
        Index(
            "uq_messages_router_status_per_run",
            "routing_run_id",
            unique=True,
            postgresql_where=text("origin IN ('router_status', 'assistant_answer')"),
        ),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    conversation_id: Mapped[UUID] = mapped_column(ForeignKey("conversations.id"), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=UTC_NOW)
    origin: Mapped[str] = mapped_column(String(32), nullable=False)
    selected_manual_ids: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    routing_run_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)

    conversation: Mapped[Conversation] = relationship(back_populates="messages", foreign_keys=[conversation_id])
    routing_run: Mapped[RoutingRun | None] = relationship(foreign_keys=[routing_run_id], post_update=True)


# --- Routing audit: policy, runs, their exact inputs, and human evaluations ---


class RoutingPolicy(Base):
    __tablename__ = "routing_policies"

    version: Mapped[str] = mapped_column(String(100), primary_key=True)
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    response_schema: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=UTC_NOW)


class RoutingRun(Base):
    """One routing attempt; status determines which decision, error, and timing fields must be set."""

    __tablename__ = "routing_runs"
    __table_args__ = (
        UniqueConstraint("id", "conversation_id", name="uq_routing_runs_id_conversation"),
        ForeignKeyConstraint(
            ["user_message_id", "conversation_id"],
            ["messages.id", "messages.conversation_id"],
            name="fk_routing_runs_user_message_same_conversation",
        ),
        CheckConstraint("status IN ('pending', 'classified', 'failed')", name="ck_routing_runs_status"),
        CheckConstraint("model_calls IN (0, 1)", name="ck_routing_runs_model_calls"),
        CheckConstraint(
            "outcome IS NULL OR outcome IN ('kb_answer', 'web_answer', 'clarification', 'blocked', 'insufficient', 'direct', 'error')",
            name="ck_routing_runs_outcome",
        ),
        CheckConstraint(
            "jsonb_typeof(citations) = 'array' AND jsonb_typeof(trace) = 'array'", name="ck_routing_runs_arrays"
        ),
        CheckConstraint("elapsed_ms IS NULL OR elapsed_ms >= 0", name="ck_routing_runs_elapsed_nonnegative"),
        CheckConstraint("routing_ms IS NULL OR routing_ms >= 0", name="ck_routing_runs_routing_nonnegative"),
        CheckConstraint("rationale IS NULL OR char_length(rationale) <= 400", name="ck_routing_runs_rationale_length"),
        CheckConstraint(
            "route IS NULL OR (route = 'retrieval' AND reason_code = 'code_lookup') OR "
            "(route = 'conversation' AND reason_code IN ('greeting', 'clarification', 'workspace_help')) OR "
            "(route = 'invalid' AND reason_code IN ('unsafe_request', 'prompt_injection', 'off_topic'))",
            name="ck_routing_runs_route_reason",
        ),
        CheckConstraint(
            "(status = 'pending' AND route IS NULL AND reason_code IS NULL AND rationale IS NULL AND error_code IS NULL AND completed_at IS NULL) OR "
            "(status = 'classified' AND route IS NOT NULL AND reason_code IS NOT NULL AND rationale IS NOT NULL AND error_code IS NULL AND completed_at IS NOT NULL) OR "
            "(status = 'failed' AND route IS NULL AND reason_code IS NULL AND rationale IS NULL AND error_code IS NOT NULL AND completed_at IS NOT NULL)",
            name="ck_routing_runs_status_fields",
        ),
        Index("ix_routing_runs_created_status", "created_at", "status"),
        Index(
            "uq_routing_runs_pending_conversation",
            "conversation_id",
            unique=True,
            postgresql_where=text("status = 'pending'"),
        ),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    conversation_id: Mapped[UUID] = mapped_column(ForeignKey("conversations.id"), nullable=False)
    user_message_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    policy_version: Mapped[str] = mapped_column(ForeignKey("routing_policies.version"), nullable=False)
    model: Mapped[str] = mapped_column(String(300), nullable=False)
    model_digest: Mapped[str | None] = mapped_column(String(300), nullable=True)
    model_options: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    route: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reason_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    rationale: Mapped[str | None] = mapped_column(String(400), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    outcome: Mapped[str | None] = mapped_column(String(32), nullable=True)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    trace: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb"))
    review: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    city: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=UTC_NOW)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    elapsed_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    routing_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    conversation: Mapped[Conversation] = relationship(back_populates="routing_runs", foreign_keys=[conversation_id])
    user_message: Mapped[Message] = relationship(foreign_keys=[user_message_id])
    policy: Mapped[RoutingPolicy] = relationship()
    context_messages: Mapped[list[RoutingRunContextMessage]] = relationship(back_populates="run")
    selected_sources: Mapped[list[RoutingRunSource]] = relationship(back_populates="run")
    evaluations: Mapped[list[RoutingEvaluation]] = relationship(back_populates="run")
    trace_events: Mapped[list[RunTraceEvent]] = relationship(back_populates="run", order_by="RunTraceEvent.seq")


class RoutingEvaluation(Base):
    """An appended human label for a run's expected route; never overwrites the model decision."""

    __tablename__ = "routing_evaluations"
    __table_args__ = (
        CheckConstraint("char_length(comment) <= 5000", name="ck_routing_evaluations_comment_length"),
        Index("ix_routing_evaluations_run_latest", "run_id", text("created_at DESC"), text("id DESC")),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("routing_runs.id"), nullable=False)
    expected_route: Mapped[str] = mapped_column(String(32), nullable=False)
    comment: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=UTC_NOW)

    run: Mapped[RoutingRun] = relationship(back_populates="evaluations")


class RoutingRunContextMessage(Base):
    """Ordered membership of a message in a run's provider transcript, within the run's conversation."""

    __tablename__ = "routing_run_context_messages"
    __table_args__ = (
        UniqueConstraint("run_id", "ordinal", name="uq_routing_run_context_ordinal"),
        UniqueConstraint("run_id", "message_id", name="uq_routing_run_context_message"),
        ForeignKeyConstraint(
            ["run_id", "conversation_id"],
            ["routing_runs.id", "routing_runs.conversation_id"],
            name="fk_routing_context_run_same_conversation",
        ),
        ForeignKeyConstraint(
            ["message_id", "conversation_id"],
            ["messages.id", "messages.conversation_id"],
            name="fk_routing_context_message_same_conversation",
        ),
        CheckConstraint("ordinal >= 0", name="ck_routing_context_ordinal_nonnegative"),
    )

    run_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    conversation_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)

    run: Mapped[RoutingRun] = relationship(back_populates="context_messages")
    message: Mapped[Message] = relationship(viewonly=True)


class RoutingRunSource(Base):
    """Ordered document version captured as a run's selected scope."""

    __tablename__ = "routing_run_sources"
    __table_args__ = (
        UniqueConstraint("run_id", "ordinal", name="uq_routing_run_sources_ordinal"),
        UniqueConstraint("run_id", "document_id", name="uq_routing_run_sources_document"),
        ForeignKeyConstraint(["run_id"], ["routing_runs.id"], name="fk_routing_sources_run"),
        ForeignKeyConstraint(
            ["source_version_id", "document_id"],
            ["document_versions.id", "document_versions.document_id"],
            name="fk_routing_sources_version_same_document",
        ),
        CheckConstraint("ordinal >= 0", name="ck_routing_sources_ordinal_nonnegative"),
    )

    run_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[str] = mapped_column(String(64), nullable=False)
    source_version_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)

    run: Mapped[RoutingRun] = relationship(back_populates="selected_sources")
    source_version: Mapped[DocumentVersion] = relationship()


class RunTraceEvent(Base):
    """One immutable structured event emitted while answering a routing run."""

    __tablename__ = "run_trace_events"
    __table_args__ = (
        UniqueConstraint("run_id", "seq", name="uq_run_trace_events_run_seq"),
        CheckConstraint("seq >= 0", name="ck_run_trace_events_seq_nonnegative"),
        CheckConstraint("duration_ms >= 0", name="ck_run_trace_events_duration_nonnegative"),
        CheckConstraint(
            "kind IN ('node', 'llm_call', 'retrieval', 'web_search', 'decision')",
            name="ck_run_trace_events_kind",
        ),
        Index("ix_run_trace_events_run_seq", "run_id", "seq"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("routing_runs.id", ondelete="RESTRICT"), nullable=False)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    node: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    data: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )

    run: Mapped[RoutingRun] = relationship(back_populates="trace_events")

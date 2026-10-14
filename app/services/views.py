"""Projections from stored rows to wire DTOs, plus the loaders they share across services."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.errors import ServiceError
from app.db.models import Conversation, Message as MessageRecord, RoutingEvaluation, RoutingRun, RunTraceEvent
from app.schemas import (
    ConversationSummary,
    Evaluation,
    Message,
    MessageResult,
    RouteDecision,
    RoutingRunView,
    TraceEvent,
)

RUN_CHILDREN = (
    selectinload(RoutingRun.context_messages),
    selectinload(RoutingRun.selected_sources),
    selectinload(RoutingRun.evaluations),
)

FAILED_ANSWER_CONTENT = "FAILED: The assistant could not complete this request. Your message was saved; reload this conversation for its recorded status."


def utcnow() -> datetime:
    return datetime.now(UTC)


def conversation_view(record: Conversation) -> ConversationSummary:
    return ConversationSummary(
        id=record.id, title=record.title, created_at=record.created_at, updated_at=record.updated_at
    )


def message_view(record: MessageRecord) -> Message:
    return Message(
        id=record.id,
        conversation_id=record.conversation_id,
        sequence=record.sequence,
        role=record.role,
        content=record.content,
        created_at=record.created_at,
        origin=record.origin,
        selected_manual_ids=record.selected_manual_ids or [],
        routing_run_id=record.routing_run_id,
    )


def evaluation_view(record: RoutingEvaluation) -> Evaluation:
    return Evaluation(
        id=record.id,
        run_id=record.run_id,
        expected_route=record.expected_route,
        comment=record.comment,
        created_at=record.created_at,
    )


def trace_event_view(record: RunTraceEvent) -> TraceEvent:
    return TraceEvent(
        seq=record.seq,
        node=record.node,
        kind=record.kind,
        started_at=record.started_at,
        duration_ms=record.duration_ms,
        summary=record.summary,
        data=record.data,
    )


def build_run_view(run: RoutingRun, question: str) -> RoutingRunView:
    """Project a run whose children are already loaded (see RUN_CHILDREN)."""
    context = sorted(run.context_messages, key=lambda item: item.ordinal)
    sources = sorted(run.selected_sources, key=lambda item: item.ordinal)
    latest = max(run.evaluations, key=lambda item: (item.created_at, item.id), default=None)
    decision = (
        RouteDecision(route=run.route, reason_code=run.reason_code, rationale=run.rationale)
        if run.status == "classified"
        else None
    )
    return RoutingRunView(
        id=run.id,
        conversation_id=run.conversation_id,
        user_message_id=run.user_message_id,
        question=question,
        status=run.status,
        decision=decision,
        outcome=run.outcome,
        citations=run.citations or [],
        trace=run.trace or [],
        review=run.review,
        city=run.city,
        error_code=run.error_code,
        created_at=run.created_at,
        completed_at=run.completed_at,
        model=run.model,
        model_digest=run.model_digest,
        policy_version=run.policy_version,
        model_options=run.model_options or {},
        elapsed_ms=run.elapsed_ms,
        routing_ms=run.routing_ms,
        model_calls=run.model_calls,
        context_message_ids=[item.message_id for item in context],
        selected_manual_ids=[item.document_id for item in sources],
        selected_source_version_ids=[item.source_version_id for item in sources],
        latest_evaluation=evaluation_view(latest) if latest else None,
    )


async def run_view(session: AsyncSession, run: RoutingRun) -> RoutingRunView:
    """Project one run, looking up the text of its user message."""
    user = await session.get(MessageRecord, run.user_message_id)
    return build_run_view(run, user.content if user else "")


async def run_views(session: AsyncSession, runs: Sequence[RoutingRun]) -> list[RoutingRunView]:
    """Project many runs, loading all their user messages in one query."""
    ids = {run.user_message_id for run in runs}
    statement = select(MessageRecord.id, MessageRecord.content).where(MessageRecord.id.in_(ids))
    questions = dict((await session.execute(statement)).all()) if ids else {}
    return [build_run_view(run, questions.get(run.user_message_id, "")) for run in runs]


async def load_conversation(session: AsyncSession, conversation_id: UUID, lock: bool = False) -> Conversation:
    statement = select(Conversation).where(Conversation.id == conversation_id)
    if lock:
        statement = statement.with_for_update()
    item = await session.scalar(statement)
    if item is None:
        raise ServiceError("conversation_not_found")
    return item


async def load_run(session: AsyncSession, run_id: UUID) -> RoutingRun:
    run = await session.scalar(select(RoutingRun).where(RoutingRun.id == run_id).options(*RUN_CHILDREN))
    if run is None:
        raise ServiceError("routing_run_not_found")
    return run


async def run_result(session: AsyncSession, run: RoutingRun) -> MessageResult:
    """The persisted user/assistant message pair and the run view for one routing run."""
    run = await load_run(session, run.id)
    messages = (
        await session.scalars(
            select(MessageRecord)
            .where((MessageRecord.id == run.user_message_id) | (MessageRecord.routing_run_id == run.id))
            .order_by(MessageRecord.sequence)
        )
    ).all()
    return MessageResult(messages=[message_view(item) for item in messages], routing_run=await run_view(session, run))

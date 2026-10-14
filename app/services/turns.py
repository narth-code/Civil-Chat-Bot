"""Persist a turn before inference, then finalize its answer and audit in a new transaction."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any, Protocol
from uuid import UUID, uuid4

import httpx
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.errors import ServiceError
from app.db.bootstrap import register_agent_policy
from app.db.models import (
    Document,
    Message as MessageRecord,
    RoutingRun,
    RoutingRunContextMessage,
    RoutingRunSource,
    RunTraceEvent,
)
from app.schemas import Citation, MessageCreate, MessageResult, RouteDecision, TraceEvent
from app.services.views import FAILED_ANSWER_CONTENT, RUN_CHILDREN, load_conversation, run_result, utcnow

MAX_REQUEST_BYTES = 12_000
DEFAULT_TITLE = "New conversation"
TITLE_LENGTH = 60
OUTCOMES = {
    "kb": "kb_answer",
    "web": "web_answer",
    "clarification": "clarification",
    "blocked": "blocked",
    "insufficient_evidence": "insufficient",
    "direct": "direct",
    "error": "error",
}


class Ask(Protocol):
    async def __call__(
        self, question: str, history: list[dict[str, str]] | None, *, scope: list[str] | None = None
    ) -> dict[str, Any]: ...


class TurnService:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        ask: Ask,
        model: str,
        reviewer_model: str,
        ollama_client: httpx.AsyncClient,
        ollama_url: str,
    ) -> None:
        self._sessions = sessions
        self._ask = ask
        self._model = model
        self._reviewer_model = reviewer_model
        self._ollama_client = ollama_client
        self._ollama_url = ollama_url.rstrip("/")

    async def send_message(self, conversation_id: UUID, payload: MessageCreate) -> MessageResult:
        async with self._sessions.begin() as session:
            conversation = await load_conversation(session, conversation_id, lock=True)
            replayed = await _replay_if_seen(session, conversation_id, payload)
            if replayed is not None:
                return replayed
            await _reject_pending_run(session, conversation_id)
            user = await _append_user_message(session, conversation, payload)
            policy = await register_agent_policy(session)
            scope = await _resolve_scope(session, payload.selected_manual_ids)
            run = RoutingRun(
                conversation_id=conversation_id,
                user_message_id=user.id,
                status="pending",
                policy_version=policy.version,
                model=self._model,
                model_digest=None,
                model_options={"reviewer_model": self._reviewer_model, "temperature": 0},
                route=None,
                reason_code=None,
                rationale=None,
                error_code=None,
                outcome=None,
                citations=[],
                trace=[],
                review=None,
                city=None,
                created_at=utcnow(),
                completed_at=None,
                elapsed_ms=None,
                routing_ms=None,
                model_calls=0,
            )
            session.add(run)
            await session.flush()
            context = await _load_context(session, conversation_id)
            _record_context_and_sources(session, run.id, conversation_id, context, scope)
            history = [{"role": item.role, "content": item.content} for item in context if item.id != user.id]
            turns = [*history, {"role": "user", "content": payload.content}]
            over_budget = len(json.dumps(turns, ensure_ascii=False).encode("utf-8")) > MAX_REQUEST_BYTES
            if over_budget:
                await _finish(session, run, error_code="context_limit", content=FAILED_ANSWER_CONTENT, model_calls=0)
                oversized = await run_result(session, run)
            run_id = run.id
        if over_budget:
            raise _turn_error("context_limit", conversation_id, payload.message_id, run_id, oversized)

        started = perf_counter()
        try:
            output = await self._ask(payload.content, history, scope=user.selected_manual_ids or None)
        except Exception as exc:
            code = _failure_code(exc)
            return await self._finalize_failure(
                conversation_id, payload.message_id, run_id, code, started, events=getattr(exc, "events", [])
            )
        return await self._finalize_output(conversation_id, payload.message_id, run_id, output, started)

    async def mark_orphaned_runs(self) -> None:
        async with self._sessions.begin() as session:
            runs = (
                await session.scalars(select(RoutingRun).where(RoutingRun.status == "pending").with_for_update())
            ).all()
            for run in runs:
                await _finish(
                    session, run, error_code="interrupted", content=FAILED_ANSWER_CONTENT, model_calls=run.model_calls
                )

    async def _digest(self) -> str | None:
        try:
            response = await self._ollama_client.get(f"{self._ollama_url}/api/tags")
            response.raise_for_status()
            for item in response.json().get("models", []):
                if item.get("name") == self._model:
                    return item.get("digest")
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            pass
        return None

    async def _finalize_failure(
        self,
        conversation_id: UUID,
        message_id: UUID,
        run_id: UUID,
        code: str,
        started: float,
        events: list[dict[str, Any]] | None = None,
    ) -> MessageResult:
        routing_ms = (perf_counter() - started) * 1000
        digest = await self._digest()
        async with self._sessions.begin() as session:
            run = await _lock_pending_run(session, conversation_id, message_id, run_id)
            await _finish(
                session,
                run,
                error_code=code,
                content=FAILED_ANSWER_CONTENT,
                model_calls=1,
                model_digest=digest,
                elapsed_ms=(perf_counter() - started) * 1000,
                routing_ms=routing_ms,
                events=events,
            )
            result = await run_result(session, run)
        raise _turn_error(code, conversation_id, message_id, run_id, result)

    async def _finalize_output(
        self,
        conversation_id: UUID,
        message_id: UUID,
        run_id: UUID,
        output: dict[str, Any],
        started: float,
    ) -> MessageResult:
        routing_ms = (perf_counter() - started) * 1000
        try:
            if not isinstance(output, dict) or output.get("source_used") not in OUTCOMES:
                raise ValueError("unknown agent outcome")
            answer = output["answer"]
            if not isinstance(answer, str) or not answer:
                raise ValueError("missing agent answer")
            outcome = OUTCOMES[output["source_used"]]
            decision = RouteDecision.model_validate(output["decision"]) if output.get("decision") is not None else None
            if outcome != "error" and decision is None:
                raise ValueError("missing router decision")
            citations = [
                Citation.model_validate(item).model_dump(exclude_none=True) for item in output.get("citations", [])
            ]
            trace = output.get("trace", [])
            review = output.get("review")
            events = [TraceEvent.model_validate(item) for item in output.get("events", [])]
            city = output.get("city")
            if not isinstance(trace, list) or any(not isinstance(item, str) for item in trace):
                raise ValueError("invalid agent trace")
            if review is not None and not isinstance(review, dict):
                raise ValueError("invalid agent review")
            if city is not None and not isinstance(city, str):
                raise ValueError("invalid agent city")
        except (KeyError, ValueError, TypeError, ValidationError):
            return await self._finalize_failure(conversation_id, message_id, run_id, "provider_invalid_output", started)
        digest = await self._digest()
        async with self._sessions.begin() as session:
            run = await _lock_pending_run(session, conversation_id, message_id, run_id)
            await _finish(
                session,
                run,
                decision=decision if outcome != "error" else None,
                error_code="provider_invalid_output" if outcome == "error" else None,
                content=answer,
                outcome=outcome,
                citations=citations,
                review=review,
                trace=trace,
                city=city,
                model_calls=1,
                model_digest=digest,
                elapsed_ms=(perf_counter() - started) * 1000,
                routing_ms=routing_ms,
                events=events,
            )
            result = await run_result(session, run)
        if outcome == "error":
            raise _turn_error("provider_invalid_output", conversation_id, message_id, run_id, result)
        return result


def _failure_code(error: Exception) -> str:
    if isinstance(error, (TimeoutError, httpx.TimeoutException)):
        return "provider_timeout"
    if isinstance(error, (ConnectionError, OSError, httpx.ConnectError, httpx.NetworkError)):
        return "provider_unavailable"
    return "provider_invalid_output"


def _turn_error(
    code: str, conversation_id: UUID, message_id: UUID, run_id: UUID, result: MessageResult
) -> ServiceError:
    return ServiceError(
        code,
        conversation_id=conversation_id,
        message_id=message_id,
        run_id=run_id,
        messages=result.messages,
        routing_run=result.routing_run,
    )


async def _replay_if_seen(session: AsyncSession, conversation_id: UUID, payload: MessageCreate) -> MessageResult | None:
    existing = await session.get(MessageRecord, payload.message_id)
    if existing is None:
        return None
    if (
        existing.conversation_id != conversation_id
        or existing.content != payload.content
        or list(existing.selected_manual_ids) != list(payload.selected_manual_ids)
    ):
        raise ServiceError("message_conflict")
    run = await session.scalar(
        select(RoutingRun).where(RoutingRun.user_message_id == payload.message_id).options(*RUN_CHILDREN)
    )
    if run is None:
        raise ServiceError("message_conflict", "This message ID cannot be routed again.")
    if run.status == "pending":
        raise ServiceError(
            "conversation_busy", conversation_id=conversation_id, message_id=payload.message_id, run_id=run.id
        )
    result = await run_result(session, run)
    if run.status == "failed":
        raise _turn_error(run.error_code or "routing_failed", conversation_id, payload.message_id, run.id, result)
    return result


async def _reject_pending_run(session: AsyncSession, conversation_id: UUID) -> None:
    pending = await session.scalar(
        select(RoutingRun.id).where(RoutingRun.conversation_id == conversation_id, RoutingRun.status == "pending")
    )
    if pending is not None:
        raise ServiceError("conversation_busy", conversation_id=conversation_id, run_id=pending)


async def _next_sequence(session: AsyncSession, conversation_id: UUID) -> int:
    highest = await session.scalar(
        select(func.coalesce(func.max(MessageRecord.sequence), -1)).where(
            MessageRecord.conversation_id == conversation_id
        )
    )
    return highest + 1


async def _append_user_message(session: AsyncSession, conversation, payload: MessageCreate) -> MessageRecord:
    sequence = await _next_sequence(session, conversation.id)
    user = MessageRecord(
        id=payload.message_id,
        conversation_id=conversation.id,
        sequence=sequence,
        role="user",
        content=payload.content,
        created_at=utcnow(),
        origin="user",
        selected_manual_ids=list(payload.selected_manual_ids),
        routing_run_id=None,
    )
    session.add(user)
    if sequence == 0 and conversation.title == DEFAULT_TITLE:
        conversation.title = " ".join(payload.content.split())[:TITLE_LENGTH]
    return user


async def _resolve_scope(session: AsyncSession, manual_ids: list[str]) -> list[dict[str, Any]]:
    if not manual_ids:
        return []
    records = (await session.scalars(select(Document).where(Document.id.in_(manual_ids)))).all()
    found = {record.id: record.current_source_version_id for record in records}
    if any(found.get(document_id) is None for document_id in manual_ids):
        raise ServiceError("manual_scope_conflict")
    return [{"document_id": document_id, "source_version_id": found[document_id]} for document_id in manual_ids]


async def _load_context(session: AsyncSession, conversation_id: UUID) -> list[MessageRecord]:
    """Load user messages and assistant replies that belong in the next turn's history."""
    rows = (
        await session.execute(
            select(MessageRecord, RoutingRun.status, RoutingRun.outcome)
            .outerjoin(RoutingRun, RoutingRun.id == MessageRecord.routing_run_id)
            .where(
                MessageRecord.conversation_id == conversation_id,
                ((MessageRecord.role == "user") | (MessageRecord.origin == "assistant_answer")),
            )
            .order_by(MessageRecord.sequence)
        )
    ).all()
    latest_user_sequence = next(
        (message.sequence for message, _, _ in reversed(rows) if message.role == "user"),
        None,
    )
    open_clarification_sequence = latest_user_sequence - 1 if latest_user_sequence is not None else None
    return [
        message
        for message, status, outcome in rows
        if message.role == "user"
        or (
            status == "classified"
            and outcome != "blocked"
            and (outcome != "clarification" or message.sequence == open_clarification_sequence)
        )
    ]


def _record_context_and_sources(
    session: AsyncSession,
    run_id: UUID,
    conversation_id: UUID,
    context: list[MessageRecord],
    scope: list[dict[str, Any]],
) -> None:
    for ordinal, message in enumerate(context):
        session.add(
            RoutingRunContextMessage(
                run_id=run_id, ordinal=ordinal, message_id=message.id, conversation_id=conversation_id
            )
        )
    for ordinal, source in enumerate(scope):
        session.add(
            RoutingRunSource(
                run_id=run_id,
                ordinal=ordinal,
                document_id=source["document_id"],
                source_version_id=source["source_version_id"],
            )
        )


async def _lock_pending_run(session: AsyncSession, conversation_id: UUID, message_id: UUID, run_id: UUID) -> RoutingRun:
    run = await session.get(RoutingRun, run_id, with_for_update=True)
    if run is None or run.status != "pending":
        raise ServiceError(
            "database_unavailable",
            "The routing audit could not be finalized.",
            conversation_id=conversation_id,
            message_id=message_id,
            run_id=run_id,
        )
    return run


async def _finish(
    session: AsyncSession,
    run: RoutingRun,
    *,
    content: str,
    error_code: str | None = None,
    decision: RouteDecision | None = None,
    outcome: str | None = None,
    citations: list[dict[str, Any]] | None = None,
    trace: list[str] | None = None,
    review: dict[str, Any] | None = None,
    city: str | None = None,
    model_calls: int = 0,
    model_digest: str | None = None,
    elapsed_ms: float | None = None,
    routing_ms: float | None = None,
    events: list[TraceEvent] | list[dict[str, Any]] | None = None,
) -> None:
    run.completed_at = utcnow()
    run.elapsed_ms = round(elapsed_ms) if elapsed_ms is not None else None
    run.routing_ms = round(routing_ms) if routing_ms is not None else None
    run.model_calls = model_calls
    run.model_digest = model_digest
    run.outcome = outcome
    run.citations = citations or []
    run.trace = trace or []
    run.review = review
    run.city = city
    if error_code is None:
        run.status = "classified"
        run.route = decision.route.value
        run.reason_code = decision.reason_code.value
        run.rationale = decision.rationale
        run.error_code = None
    else:
        run.status = "failed"
        run.route = run.reason_code = run.rationale = None
        run.error_code = error_code
    run.city = city
    for seq, event in enumerate(events or []):
        parsed = event if isinstance(event, TraceEvent) else TraceEvent.model_validate(event)
        session.add(
            RunTraceEvent(
                run_id=run.id,
                seq=seq,
                node=parsed.node,
                kind=parsed.kind,
                started_at=parsed.started_at,
                duration_ms=parsed.duration_ms,
                summary=parsed.summary,
                data=parsed.data,
            )
        )
    conversation = await load_conversation(session, run.conversation_id, lock=True)
    sequence = await _next_sequence(session, run.conversation_id)
    user = await session.get(MessageRecord, run.user_message_id)
    session.add(
        MessageRecord(
            id=uuid4(),
            conversation_id=run.conversation_id,
            sequence=sequence,
            role="assistant",
            content=content,
            created_at=utcnow(),
            origin="assistant_answer",
            selected_manual_ids=list(user.selected_manual_ids) if user else [],
            routing_run_id=run.id,
        )
    )
    conversation.updated_at = utcnow()
    await session.flush()

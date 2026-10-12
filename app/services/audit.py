"""Read side of routing: windowed metrics, run history and detail, and human route evaluations."""

from __future__ import annotations

from datetime import timedelta
from uuid import UUID

from sqlalchemy import ColumnElement, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.errors import ServiceError
from app.db.models import Message as MessageRecord, RoutingEvaluation, RoutingPolicy, RoutingRun, RunTraceEvent
from app.schemas import (
    Evaluation,
    EvaluationCreate,
    PolicyView,
    RoutingDetail,
    RoutingList,
    RoutingRunTrace,
    RoutingWindow,
)
from app.services.views import (
    RUN_CHILDREN,
    evaluation_view,
    load_run,
    message_view,
    run_view,
    run_views,
    trace_event_view,
    utcnow,
)
from app.services.metrics import routing_metrics, window_filter


class AuditService:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def routing_list(self, days: int, offset: int, limit: int) -> RoutingList:
        """Metrics for the trailing window plus one newest-first page of its runs."""
        end = utcnow()
        start = end - timedelta(days=days)
        async with self._sessions() as session:
            metrics = await routing_metrics(session, start, end)
            runs = await _run_page(session, window_filter(start, end), offset, limit)
            return RoutingList(
                window=RoutingWindow(days=days, start=start, end=end),
                metrics=metrics,
                items=await run_views(session, runs),
                total=metrics.total,
                offset=offset,
                limit=limit,
            )

    async def routing_detail(self, run_id: UUID) -> RoutingDetail:
        async with self._sessions() as session:
            run = await load_run(session, run_id)
            contexts = sorted(run.context_messages, key=lambda item: item.ordinal)
            message_ids = [item.message_id for item in contexts]
            messages = (
                (await session.scalars(select(MessageRecord).where(MessageRecord.id.in_(message_ids)))).all()
                if message_ids
                else []
            )
            by_id = {item.id: item for item in messages}
            policy = await session.get(RoutingPolicy, run.policy_version)
            if policy is None:
                raise ServiceError("database_unavailable", "The recorded routing policy is unavailable.")
            return RoutingDetail(
                routing_run=await run_view(session, run),
                context_messages=[message_view(by_id[item_id]) for item_id in message_ids],
                policy=PolicyView(
                    version=policy.version, system_prompt=policy.system_prompt, response_schema=policy.response_schema
                ),
                evaluations=[
                    evaluation_view(item)
                    for item in sorted(run.evaluations, key=lambda item: (item.created_at, item.id))
                ],
            )

    async def routing_trace(self, run_id: UUID) -> RoutingRunTrace:
        async with self._sessions() as session:
            exists = await session.get(RoutingRun, run_id)
            if exists is None:
                raise ServiceError("routing_run_not_found")
            events = (
                await session.scalars(
                    select(RunTraceEvent).where(RunTraceEvent.run_id == run_id).order_by(RunTraceEvent.seq)
                )
            ).all()
            return RoutingRunTrace(run_id=run_id, events=[trace_event_view(event) for event in events])

    async def add_evaluation(self, run_id: UUID, payload: EvaluationCreate) -> Evaluation:
        async with self._sessions.begin() as session:
            run = await session.get(RoutingRun, run_id)
            if run is None:
                raise ServiceError("routing_run_not_found")
            evaluation = RoutingEvaluation(
                run_id=run_id, expected_route=payload.expected_route.value, comment=payload.comment, created_at=utcnow()
            )
            session.add(evaluation)
            await session.flush()
            return evaluation_view(evaluation)


async def _run_page(
    session: AsyncSession, window: tuple[ColumnElement[bool], ...], offset: int, limit: int
) -> list[RoutingRun]:
    """One page of runs in the window, newest first with id as the stable tie-break."""
    statement = (
        select(RoutingRun)
        .where(*window)
        .options(*RUN_CHILDREN)
        .order_by(RoutingRun.created_at.desc(), RoutingRun.id.desc())
        .offset(offset)
        .limit(limit)
    )
    return list((await session.scalars(statement)).all())

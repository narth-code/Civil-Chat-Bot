"""Windowed aggregate queries over routing runs, and their assembly into RoutingMetrics."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import ColumnElement, func, literal_column, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import RoutingEvaluation, RoutingRun
from app.schemas import EvaluationMetrics, LatencyMetrics, Route, RoutingMetrics


def window_filter(start: datetime, end: datetime) -> tuple[ColumnElement[bool], ...]:
    """Runs created in the half-open window [start, end)."""
    return (RoutingRun.created_at >= start, RoutingRun.created_at < end)


async def status_counts(session: AsyncSession, window: tuple[ColumnElement[bool], ...]) -> dict[str, int]:
    """Number of runs in the window per status; absent statuses are omitted."""
    statement = select(RoutingRun.status, func.count()).where(*window).group_by(RoutingRun.status)
    return {status: count for status, count in (await session.execute(statement)).all()}


async def route_counts(session: AsyncSession, window: tuple[ColumnElement[bool], ...]) -> dict[Route, int]:
    """Classified runs per route, with every route present."""
    counts = {route: 0 for route in Route}
    statement = (
        select(RoutingRun.route, func.count())
        .where(*window, RoutingRun.status == "classified")
        .group_by(RoutingRun.route)
    )
    for route, count in (await session.execute(statement)).all():
        counts[Route(route)] = count
    return counts


async def invalid_reason_counts(session: AsyncSession, window: tuple[ColumnElement[bool], ...]) -> dict[str, int]:
    """Runs routed to invalid, per reason code."""
    statement = (
        select(RoutingRun.reason_code, func.count())
        .where(*window, RoutingRun.route == "invalid")
        .group_by(RoutingRun.reason_code)
    )
    return {reason: count for reason, count in (await session.execute(statement)).all() if reason is not None}


async def latency_metrics(session: AsyncSession, window: tuple[ColumnElement[bool], ...]) -> LatencyMetrics:
    """Median and p95 routing time over completed generation attempts."""
    statement = select(
        func.count(),
        func.percentile_cont(literal_column("ARRAY[0.5,0.95]")).within_group(RoutingRun.routing_ms),
    ).where(
        *window,
        RoutingRun.status != "pending",
        RoutingRun.model_calls == 1,
        RoutingRun.routing_ms.is_not(None),
    )
    sample_count, quantiles = (await session.execute(statement)).one()
    return LatencyMetrics(
        sample_count=sample_count,
        median_ms=quantiles[0] if quantiles else None,
        p95_ms=quantiles[1] if quantiles else None,
    )


async def evaluation_metrics(session: AsyncSession, window: tuple[ColumnElement[bool], ...]) -> EvaluationMetrics:
    """Agreement between classified decisions and each run's latest human label."""
    latest_first = (RoutingEvaluation.created_at.desc(), RoutingEvaluation.id.desc())
    ranked = select(
        RoutingEvaluation.run_id,
        RoutingEvaluation.expected_route,
        func.row_number().over(partition_by=RoutingEvaluation.run_id, order_by=latest_first).label("rank"),
    ).subquery()
    statement = (
        select(RoutingRun.route, ranked.c.expected_route)
        .join(ranked, ranked.c.run_id == RoutingRun.id)
        .where(*window, RoutingRun.status == "classified", ranked.c.rank == 1)
    )
    rows = (await session.execute(statement)).all()
    agreements = sum(1 for route, expected in rows if route == expected)
    return EvaluationMetrics(
        reviewed_count=len(rows),
        agreement_count=agreements,
        agreement_rate=agreements / len(rows) if rows else None,
    )


async def outcome_counts(session: AsyncSession, window: tuple[ColumnElement[bool], ...]) -> dict[str, int]:
    counts = {
        name: 0 for name in ("kb_answer", "web_answer", "clarification", "blocked", "insufficient", "direct", "error")
    }
    statement = (
        select(RoutingRun.outcome, func.count())
        .where(*window, RoutingRun.outcome.is_not(None))
        .group_by(RoutingRun.outcome)
    )
    for outcome, count in (await session.execute(statement)).all():
        counts[outcome] = count
    return counts


async def routing_metrics(session: AsyncSession, start: datetime, end: datetime) -> RoutingMetrics:
    """All routing metrics for runs created in [start, end)."""
    window = window_filter(start, end)
    statuses = await status_counts(session, window)
    return RoutingMetrics(
        total=sum(statuses.values()),
        pending=statuses.get("pending", 0),
        failed=statuses.get("failed", 0),
        classified=statuses.get("classified", 0),
        route_counts=await route_counts(session, window),
        invalid_reason_counts=await invalid_reason_counts(session, window),
        latency=await latency_metrics(session, window),
        evaluation=await evaluation_metrics(session, window),
        outcome_counts=await outcome_counts(session, window),
    )

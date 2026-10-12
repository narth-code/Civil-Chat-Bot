"""Structured, append-only execution traces for graph invocations."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Callable

MAX_TEXT_CHARS = 20_000


def truncate(value: Any, limit: int = MAX_TEXT_CHARS) -> tuple[str, bool]:
    """Return text suitable for trace storage and whether it was shortened."""
    text = value if isinstance(value, str) else str(value)
    return (text[:limit], len(text) > limit)


def add_event(
    state: dict[str, Any],
    node: str,
    kind: str,
    summary: str,
    started_at: datetime | None = None,
    **data: Any,
) -> list[dict[str, Any]]:
    """Append one serialisable event without mutating the graph state."""
    started = started_at or datetime.now(UTC)
    return [
        *state.get("events", []),
        {
            "node": node,
            "kind": kind,
            "started_at": started.isoformat(),
            "duration_ms": max(0, int(data.pop("duration_ms", 0))),
            "summary": summary,
            "data": data,
        },
    ]


def traced(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a graph node and append a timing event while preserving its update."""

    async def invoke(state: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        started_at = datetime.now(UTC)
        started = perf_counter()
        try:
            result = fn(state, *args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
        except Exception as error:
            events = add_event(
                state,
                name,
                "node",
                f"{name} -> error: {type(error).__name__}",
                started_at,
                duration_ms=round((perf_counter() - started) * 1000),
                error=type(error).__name__,
            )
            # LangGraph cannot merge an update from a failed node; carry it on the exception to persistence.
            state["events"] = events
            setattr(error, "events", events)
            raise
        update = dict(result or {})
        summaries = update.get("trace", state.get("trace", []))
        prior = state.get("trace", [])
        produced = summaries[len(prior) :] if isinstance(summaries, list) else []
        summary = produced[-1] if produced else name
        event_state = {**state, "events": update.get("events", state.get("events", []))}
        update["events"] = add_event(
            event_state,
            name,
            "node",
            summary,
            started_at,
            duration_ms=round((perf_counter() - started) * 1000),
        )
        return update

    return invoke

"""Print persisted agent execution traces for local investigation."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.db.models import RoutingRun
from app.db.session import create_resources


async def _runs(run_id: UUID | None, last: int | None) -> Sequence[RoutingRun]:
    engine, sessions = create_resources(get_settings().database_url)
    try:
        async with sessions() as session:
            statement = select(RoutingRun).options(
                selectinload(RoutingRun.trace_events), selectinload(RoutingRun.user_message)
            )
            if run_id is not None:
                statement = statement.where(RoutingRun.id == run_id)
            else:
                statement = statement.order_by(RoutingRun.created_at.desc(), RoutingRun.id.desc()).limit(last or 1)
            return (await session.scalars(statement)).all()
    finally:
        await engine.dispose()


def _heading(run: RoutingRun) -> str:
    question = run.user_message.content if run.user_message else ""
    return f"{run.id} outcome={run.outcome or run.status} city={run.city or '-'} question={question}"


def _print(run: RoutingRun, verbose: bool) -> None:
    print(_heading(run))
    for event in sorted(run.trace_events, key=lambda item: item.seq):
        print(f"  {event.seq} {event.node} {event.kind} {event.duration_ms}ms {event.summary}")
        if verbose:
            print(f"    {json.dumps(event.data, ensure_ascii=False, sort_keys=True)}")


async def main(run_id: UUID | None, last: int | None, verbose: bool) -> int:
    runs = await _runs(run_id, last)
    if not runs:
        print("No matching routing runs.")
        return 1
    for index, run in enumerate(runs):
        if index:
            print()
        _print(run, verbose)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m app.rag.trace_report")
    parser.add_argument("run_id", type=UUID, nargs="?")
    parser.add_argument("--last", type=int, metavar="N")
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args()
    if arguments.last is not None and arguments.last < 1:
        parser.error("--last must be positive")
    if arguments.run_id is not None and arguments.last is not None:
        parser.error("run_id and --last cannot be used together")
    raise SystemExit(asyncio.run(main(arguments.run_id, arguments.last, arguments.verbose)))

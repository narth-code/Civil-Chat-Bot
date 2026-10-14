"""Conversation listing, creation, detail, and one-time import of browser history."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.errors import ServiceError
from app.db.models import Conversation, Message as MessageRecord, RoutingRun
from app.schemas import ConversationDetail, ConversationImport, ConversationSummary, ImportResult
from app.services.views import RUN_CHILDREN, conversation_view, load_conversation, message_view, run_view, utcnow


class ConversationService:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def list_conversations(self) -> list[ConversationSummary]:
        async with self._sessions() as session:
            rows = (
                await session.scalars(
                    select(Conversation).order_by(Conversation.updated_at.desc(), Conversation.id.desc())
                )
            ).all()
            return [conversation_view(row) for row in rows]

    async def create_conversation(self) -> ConversationSummary:
        now = utcnow()
        record = Conversation(title="New conversation", created_at=now, updated_at=now)
        async with self._sessions.begin() as session:
            session.add(record)
        return conversation_view(record)

    async def conversation_detail(self, conversation_id: UUID) -> ConversationDetail:
        async with self._sessions() as session:
            record = await load_conversation(session, conversation_id)
            messages = (
                await session.scalars(
                    select(MessageRecord)
                    .where(MessageRecord.conversation_id == conversation_id)
                    .order_by(MessageRecord.sequence)
                )
            ).all()
            run = await session.scalar(
                select(RoutingRun)
                .where(RoutingRun.conversation_id == conversation_id, RoutingRun.status == "pending")
                .options(*RUN_CHILDREN)
            )
            pending = await run_view(session, run) if run else None
            return ConversationDetail(
                conversation=conversation_view(record),
                messages=[message_view(message) for message in messages],
                pending_run=pending,
            )

    async def import_conversations(self, payload: ConversationImport) -> ImportResult:
        async with self._sessions.begin() as session:
            for incoming in payload.conversations:
                existing = await session.get(Conversation, incoming.id, with_for_update=True)
                if existing is None:
                    latest = max((item.created_at for item in incoming.messages), default=incoming.created_at)
                    existing = Conversation(
                        id=incoming.id, title=incoming.title, created_at=incoming.created_at, updated_at=latest
                    )
                    session.add(existing)
                    for sequence, item in enumerate(incoming.messages):
                        session.add(
                            MessageRecord(
                                id=item.id,
                                conversation_id=incoming.id,
                                sequence=sequence,
                                role=item.role,
                                content=item.content,
                                created_at=item.created_at,
                                origin="legacy_demo" if item.role == "assistant" else "user",
                                selected_manual_ids=list(item.selected_manual_ids),
                                routing_run_id=None,
                            )
                        )
                    continue
                current = (
                    await session.scalars(
                        select(MessageRecord)
                        .where(MessageRecord.conversation_id == incoming.id)
                        .order_by(MessageRecord.sequence)
                    )
                ).all()
                if not _same_import(existing, current, incoming):
                    raise ServiceError("import_conflict")
        return ImportResult(imported_conversation_ids=[item.id for item in payload.conversations])


def _same_import(existing: Conversation, current: list[MessageRecord], incoming: Any) -> bool:
    if (
        existing.title != incoming.title
        or existing.created_at != incoming.created_at
        or len(current) != len(incoming.messages)
    ):
        return False
    return all(
        record.id == item.id
        and record.sequence == index
        and record.role == item.role
        and record.content == item.content
        and record.created_at == item.created_at
        and list(record.selected_manual_ids or []) == list(item.selected_manual_ids)
        and record.origin == ("legacy_demo" if item.role == "assistant" else "user")
        for index, (record, item) in enumerate(zip(current, incoming.messages))
    )

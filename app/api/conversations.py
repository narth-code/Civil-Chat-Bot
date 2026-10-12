"""Conversation, import, and message-send routes."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends

from app.api.deps import conversations as conversations_service, turns as turns_service
from app.schemas import (
    ConversationCreate,
    ConversationDetail,
    ConversationImport,
    ConversationList,
    ConversationSummary,
    ImportResult,
    MessageCreate,
    MessageResult,
)
from app.services.conversations import ConversationService
from app.services.turns import TurnService

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("", response_model=ConversationList)
async def list_conversations(conversations: ConversationService = Depends(conversations_service)) -> ConversationList:
    return ConversationList(conversations=await conversations.list_conversations())


@router.post("", response_model=ConversationSummary)
async def create_conversation(
    _: ConversationCreate, conversations: ConversationService = Depends(conversations_service)
) -> ConversationSummary:
    return await conversations.create_conversation()


@router.post("/import", response_model=ImportResult)
async def import_conversations(
    payload: ConversationImport, conversations: ConversationService = Depends(conversations_service)
) -> ImportResult:
    return await conversations.import_conversations(payload)


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: UUID, conversations: ConversationService = Depends(conversations_service)
) -> ConversationDetail:
    return await conversations.conversation_detail(conversation_id)


@router.post("/{conversation_id}/messages", response_model=MessageResult)
async def send_message(
    conversation_id: UUID, payload: MessageCreate, turns: TurnService = Depends(turns_service)
) -> MessageResult:
    return await turns.send_message(conversation_id, payload)

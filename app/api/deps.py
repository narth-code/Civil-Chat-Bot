"""FastAPI dependencies that fetch each service from application state."""

from __future__ import annotations

from fastapi import Request

from app.services.audit import AuditService
from app.services.conversations import ConversationService
from app.services.turns import TurnService


def conversations(request: Request) -> ConversationService:
    return request.app.state.conversations


def turns(request: Request) -> TurnService:
    return request.app.state.turns


def audit(request: Request) -> AuditService:
    return request.app.state.audit

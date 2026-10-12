"""FastAPI composition and lifespan-owned local resources."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI

from app.api.conversations import router as conversations_router
from app.api.errors import register_error_handlers
from app.api.health import router as health_router
from app.api.manuals import router as manuals_router
from app.api.routing import router as routing_router
from app.core.config import Settings, get_settings
from app.db.session import create_resources
from app.rag.agent import ask as default_ask
from app.services.audit import AuditService
from app.services.conversations import ConversationService
from app.services.turns import TurnService

logger = logging.getLogger(__name__)


def _preload_agent() -> None:
    from app.rag.agent import embedder, llm, reviewer_llm

    for loader in (embedder, llm, reviewer_llm):
        try:
            loader()
        except Exception:
            logger.exception("Agent preload failed: %s", loader.__name__)


async def _preload_background() -> None:
    try:
        await asyncio.to_thread(_preload_agent)
    except Exception:
        logger.exception("Agent background preload failed")


def create_app(
    settings: Settings,
    *,
    ask: Callable[[str, list[dict[str, str]] | None], Awaitable[dict[str, Any]]] | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine, session_factory = create_resources(settings.database_url)
        client = httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=5.0))
        app.state.settings = settings
        app.state.engine = engine
        app.state.session_factory = session_factory
        app.state.ollama_client = client
        app.state.conversations = ConversationService(session_factory)
        app.state.turns = TurnService(
            session_factory,
            ask or default_ask,
            settings.ollama_model,
            settings.reviewer_model,
            client,
            settings.ollama_url,
        )
        app.state.audit = AuditService(session_factory)
        preload = None
        try:
            await app.state.turns.mark_orphaned_runs()
            if ask is None:
                preload = asyncio.create_task(_preload_background())
            yield
        finally:
            if preload is not None:
                preload.cancel()
            await client.aclose()
            await engine.dispose()

    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
    register_error_handlers(app)
    app.include_router(health_router)
    app.include_router(conversations_router)
    app.include_router(routing_router)
    app.include_router(manuals_router)
    return app


app = create_app(get_settings())

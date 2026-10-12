"""Readiness of the database and the configured local model."""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

router = APIRouter(tags=["health"])


@router.get("/api/health")
async def health(request: Request) -> JSONResponse:
    settings = request.app.state.settings
    database_ready = False
    model_ready = False
    try:
        async with request.app.state.session_factory() as session:
            await session.execute(text("SELECT 1"))
        database_ready = True
    except Exception:
        pass
    try:
        response = await request.app.state.ollama_client.get(f"{settings.ollama_url.rstrip('/')}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_ready = any(item.get("name") == settings.ollama_model for item in models)
    except (httpx.HTTPError, ValueError, TypeError):
        pass
    if database_ready and model_ready:
        return JSONResponse({"status": "ok", "database": "ready", "model": settings.ollama_model})
    return JSONResponse(
        status_code=503,
        content={
            "status": "unavailable",
            "database": "ready" if database_ready else "unavailable",
            "model": settings.ollama_model if model_ready else "unavailable",
        },
    )

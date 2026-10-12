"""Async SQLAlchemy resource creation for request-scoped units of work."""

from __future__ import annotations

from sqlalchemy import URL
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


def create_resources(url: str | URL) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    """Create a pool and session factory without opening a connection."""
    engine = create_async_engine(url, pool_pre_ping=True)
    return engine, async_sessionmaker(engine, expire_on_commit=False)

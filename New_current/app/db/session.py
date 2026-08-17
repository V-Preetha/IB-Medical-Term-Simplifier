"""Async SQLAlchemy engine/session wiring.

`create_engine_and_sessionmaker` is the composition-root factory (mirrors
`app/entity_linking/runtime.py`'s `create_entity_linking_service()`):
callers construct one engine/sessionmaker pair at application startup and
pass it down, rather than importing a module-level singleton engine. This
keeps tests able to point at an isolated database without patching globals.

`get_session` is an async context manager suitable for direct use in
services/tests and by the production infrastructure composition root.
"""

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db.config import DatabaseSettings


def create_engine_and_sessionmaker(
    settings: DatabaseSettings,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    """Build a new async engine and session factory from validated settings."""

    settings.validate()
    engine = create_async_engine(
        settings.database_url,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        pool_timeout=settings.pool_timeout,
        echo=settings.echo,
    )
    sessionmaker = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    return engine, sessionmaker


@asynccontextmanager
async def get_session(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Yield a session bound to one unit of work; rolls back on error."""

    async with sessionmaker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def dispose_engine(engine: AsyncEngine) -> None:
    """Release pooled connections; call during application shutdown."""

    await engine.dispose()


async def session_dependency(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """FastAPI-dependency-shaped generator for an injected session factory."""

    async with get_session(sessionmaker) as session:
        yield session

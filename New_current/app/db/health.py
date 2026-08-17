"""Database health/readiness check.

Fails closed: any connectivity error is caught and reported as unhealthy
rather than raised past the caller or silently swallowed into a fake
success. The production infrastructure health endpoint uses this probe.
"""

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine


@dataclass(frozen=True, slots=True)
class DatabaseHealth:
    """Result of a database connectivity/pool check."""

    healthy: bool
    detail: str
    pool_size: int | None
    checked_out_connections: int | None


async def check_database_health(engine: AsyncEngine) -> DatabaseHealth:
    """Run `SELECT 1` against the engine and report pool metrics.

    Returns an unhealthy result (never raises) when the database is
    unreachable, so callers can build a readiness response without a
    try/except at every call site.
    """

    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        return DatabaseHealth(
            healthy=False,
            detail=f"database unreachable: {exc.__class__.__name__}",
            pool_size=None,
            checked_out_connections=None,
        )

    pool = engine.pool
    size = pool.size() if hasattr(pool, "size") else None
    checked_out = pool.checkedout() if hasattr(pool, "checkedout") else None
    return DatabaseHealth(
        healthy=True,
        detail="database reachable",
        pool_size=size,
        checked_out_connections=checked_out,
    )

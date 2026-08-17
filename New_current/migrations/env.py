"""Alembic environment configured for the async SQLAlchemy engine.

The connection string is never hard-coded here or in `alembic.ini`. It is
read from `app.db.config.DatabaseSettings.from_environment()` so it goes
through the same validated, typed parsing the application uses at runtime.
If constructing `DatabaseSettings` fails (for example when only a bare
`DATABASE_URL` is exported and the rest of the application settings module
is not importable in a throwaway shell), this falls back to reading
`DATABASE_URL` directly from the environment -- but it never falls back to
a hard-coded or sqlite default for a real migration run, so a missing
configuration fails loudly instead of silently migrating the wrong
database.
"""

import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Make `app` importable when Alembic is invoked from `New_current/`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.base import Base  # noqa: E402
from app.db.models import (  # noqa: E402,F401
    AuditLog,
    EmbeddingRecord,
    EntityLink,
    Feedback,
    MedicalEntity,
    ModelOutput,
    ModelRegistry,
    ProcessingJob,
    Report,
    ReportProcessing,
    Simplification,
    SupportedDialect,
    User,
    UserPreference,
    VoiceGeneration,
    VoiceProfile,
)

# this is the Alembic Config object, which provides access to values within
# the .ini file in use.
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _resolve_database_url() -> str:
    """Resolve the migration target URL from validated application settings,
    falling back to the raw `DATABASE_URL` environment variable only so the
    bare `alembic` CLI works without importing the full application config
    module. Never falls back to a hard-coded credential or sqlite URL."""

    try:
        from app.db.config import DatabaseSettings

        return DatabaseSettings.from_environment().database_url
    except Exception:
        url = os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError(
                "DATABASE_URL is not set and app.db.config.DatabaseSettings "
                "could not be constructed; Alembic cannot determine the "
                "migration target."
            ) from None
        return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emits SQL, no live connection)."""

    url = _resolve_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Build an `AsyncEngine` from `alembic.ini` overridden with the
    resolved `DATABASE_URL`, connect once, and run migrations in 'online'
    mode."""

    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _resolve_database_url()
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

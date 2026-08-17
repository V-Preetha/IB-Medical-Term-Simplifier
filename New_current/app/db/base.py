"""Declarative base and shared column mixins for the persistence layer.

`TimestampVersionMixin` gives every table `created_at`, `updated_at`, and an
optimistic-concurrency `version` counter. `updated_at` is refreshed by the
database (`onupdate=func.now()`); `version` is NOT auto-incremented by the
database because portable triggers would require per-dialect DDL. Instead,
repository update paths in `app/db/repositories` are responsible for setting
`instance.version = instance.version + 1` before flush. This keeps version
bookkeeping visible in Python and testable without relying on Postgres-only
trigger syntax.

`SoftDeleteMixin` adds a nullable `deleted_at` column used by aggregates that
support soft delete (see `app/db/models.py` for which tables include it).
`GUID` is a portable UUID column type: it stores a native `UUID` on
PostgreSQL and a 32-character hex string on other dialects (used only by the
sqlite structural test fallback, never in production).
"""

import uuid
from datetime import datetime

from sqlalchemy import CHAR, DateTime, Integer
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, declared_attr, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.types import TypeDecorator


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models in this package."""


class GUID(TypeDecorator):
    """Platform-independent UUID type.

    Uses PostgreSQL's native UUID type when available, otherwise stores a
    32-character hex string. Production deployments always run against
    PostgreSQL; the string fallback exists only so model metadata and
    structural tests can run against sqlite in this sandbox.
    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(32))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
        if not isinstance(value, uuid.UUID):
            value = uuid.UUID(str(value))
        return value.hex

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(value)


class TimestampVersionMixin:
    """Adds `created_at`, `updated_at`, and an app-managed `version` column."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")

    @declared_attr.directive
    def __mapper_args__(cls) -> dict:
        return {"version_id_col": cls.version}


class SoftDeleteMixin:
    """Adds a nullable `deleted_at` column for aggregates supporting soft delete."""

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

"""Environment-backed configuration for the database layer.

Follows the frozen-dataclass `Settings.from_environment()` convention used by
`app/config.py`, `app/entity_linking/config.py`, and `app/embeddings/config.py`.
No pydantic-settings/BaseSettings dependency is introduced; values come from
plain environment variables with explicit, typed parsing and validation.
"""

import os
from dataclasses import dataclass

from app.db.errors import DatabaseConfigurationError


def _string(name: str, default: str) -> str:
    value = os.getenv(name, default)
    if not value or not value.strip():
        raise DatabaseConfigurationError(f"{name} must not be empty")
    return value.strip()


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise DatabaseConfigurationError(f"{name} must be an integer") from exc
    if value <= 0:
        raise DatabaseConfigurationError(f"{name} must be positive")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class DatabaseSettings:
    """Typed, validated configuration for the async SQLAlchemy engine."""

    database_url: str
    pool_size: int
    max_overflow: int
    pool_timeout: int
    echo: bool

    @classmethod
    def from_environment(cls) -> "DatabaseSettings":
        return cls(
            database_url=_string("DATABASE_URL", ""),
            pool_size=_positive_int("DATABASE_POOL_SIZE", 5),
            max_overflow=_positive_int("DATABASE_MAX_OVERFLOW", 10),
            pool_timeout=_positive_int("DATABASE_POOL_TIMEOUT_SECONDS", 30),
            echo=_boolean("DATABASE_ECHO", False),
        )

    def validate(self) -> None:
        if "+asyncpg" not in self.database_url and not self.database_url.startswith(
            "sqlite+aiosqlite"
        ):
            raise DatabaseConfigurationError(
                "DATABASE_URL must use the asyncpg driver (postgresql+asyncpg://...) in production"
            )

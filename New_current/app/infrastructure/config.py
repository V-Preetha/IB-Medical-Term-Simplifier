"""Environment-backed configuration for PostgreSQL, Redis, and Celery."""

import os
from dataclasses import dataclass

from cryptography.fernet import Fernet

from app.infrastructure.errors import InfrastructureConfigurationError


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
        raise InfrastructureConfigurationError(f"{name} must be a boolean.")
    return normalized in {"true", "1", "yes", "on"}


def _integer(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise InfrastructureConfigurationError(f"{name} must be an integer.") from exc
    if value < minimum:
        raise InfrastructureConfigurationError(f"{name} must be at least {minimum}.")
    return value


def _value(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True, slots=True)
class InfrastructureSettings:
    """Validated deployment settings for shared infrastructure."""

    enabled: bool
    redis_url: str
    redis_prefix: str
    cache_ttl_seconds: int
    lock_ttl_seconds: int
    cache_encryption_key: str
    celery_broker_url: str
    celery_result_backend: str
    celery_default_queue: str
    celery_task_timeout_seconds: int
    celery_max_retries: int
    celery_retry_backoff_seconds: int
    health_timeout_seconds: int

    @classmethod
    def from_environment(cls) -> "InfrastructureSettings":
        redis_url = _value("REDIS_URL", "redis://localhost:6379/0")
        return cls(
            enabled=_boolean("INFRASTRUCTURE_ENABLED", False),
            redis_url=redis_url,
            redis_prefix=_value("REDIS_KEY_PREFIX", "ibhealth"),
            cache_ttl_seconds=_integer("CACHE_TTL_SECONDS", 3600),
            lock_ttl_seconds=_integer("CACHE_LOCK_TTL_SECONDS", 300),
            cache_encryption_key=_value("CACHE_ENCRYPTION_KEY"),
            celery_broker_url=_value("CELERY_BROKER_URL", redis_url),
            celery_result_backend=_value("CELERY_RESULT_BACKEND", redis_url),
            celery_default_queue=_value("CELERY_DEFAULT_QUEUE", "ibhealth.cpu"),
            celery_task_timeout_seconds=_integer("CELERY_TASK_TIMEOUT_SECONDS", 900),
            celery_max_retries=_integer("CELERY_MAX_RETRIES", 3, minimum=0),
            celery_retry_backoff_seconds=_integer("CELERY_RETRY_BACKOFF_SECONDS", 5),
            health_timeout_seconds=_integer("INFRASTRUCTURE_HEALTH_TIMEOUT_SECONDS", 3),
        )

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.redis_url.startswith(("redis://", "rediss://")):
            raise InfrastructureConfigurationError("REDIS_URL must use redis:// or rediss://.")
        if not self.redis_prefix:
            raise InfrastructureConfigurationError("REDIS_KEY_PREFIX must not be blank.")
        if not self.cache_encryption_key:
            raise InfrastructureConfigurationError(
                "CACHE_ENCRYPTION_KEY is required when infrastructure is enabled."
            )
        try:
            Fernet(self.cache_encryption_key.encode("ascii"))
        except (ValueError, UnicodeEncodeError) as exc:
            raise InfrastructureConfigurationError(
                "CACHE_ENCRYPTION_KEY must be a valid Fernet key."
            ) from exc

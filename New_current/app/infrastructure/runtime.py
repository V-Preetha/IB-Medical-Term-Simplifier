"""Application lifecycle composition for production infrastructure."""

from datetime import UTC, datetime

from redis.asyncio import Redis

from app.db.config import DatabaseSettings
from app.db.session import create_engine_and_sessionmaker, dispose_engine
from app.infrastructure.cache import CacheKeyBuilder, RedisStageCache
from app.infrastructure.celery_app import create_celery_app
from app.infrastructure.config import InfrastructureSettings
from app.infrastructure.errors import InfrastructureConfigurationError
from app.infrastructure.queue import CeleryJobQueue
from app.infrastructure.schemas import (
    ComponentHealth,
    InfrastructureHealthResponse,
    InfrastructureMetrics,
)
from app.infrastructure.service import InfrastructureService


class InfrastructureRuntime:
    """Own shared clients and expose a fail-closed optional local startup mode."""

    def __init__(self, settings: InfrastructureSettings | None = None) -> None:
        self.settings = settings or InfrastructureSettings.from_environment()
        self._engine = None
        self._cache: RedisStageCache | None = None
        self._service: InfrastructureService | None = None

    @property
    def service(self) -> InfrastructureService:
        if self._service is None:
            raise InfrastructureConfigurationError(
                "Production infrastructure is not enabled for this process."
            )
        return self._service

    async def initialize(self) -> None:
        self.settings.validate()
        if not self.settings.enabled:
            return
        engine, sessions = create_engine_and_sessionmaker(DatabaseSettings.from_environment())
        redis = Redis.from_url(
            self.settings.redis_url,
            decode_responses=False,
            socket_connect_timeout=self.settings.health_timeout_seconds,
            socket_timeout=self.settings.health_timeout_seconds,
            health_check_interval=30,
        )
        cache = RedisStageCache(
            redis,
            key_builder=CacheKeyBuilder(self.settings.redis_prefix),
            encryption_key=self.settings.cache_encryption_key,
            default_ttl_seconds=self.settings.cache_ttl_seconds,
            lock_ttl_seconds=self.settings.lock_ttl_seconds,
        )
        queue = CeleryJobQueue(
            create_celery_app(self.settings),
            redis,
            health_timeout=self.settings.health_timeout_seconds,
        )
        self._engine = engine
        self._cache = cache
        self._service = InfrastructureService(
            engine=engine,
            session_factory=sessions,
            cache=cache,
            queue=queue,
        )

    async def health(self) -> InfrastructureHealthResponse:
        if self._service is not None:
            return await self._service.health()
        detail = "Set INFRASTRUCTURE_ENABLED=true and configure deployment services."
        component = ComponentHealth(status="not_configured", detail=detail)
        return InfrastructureHealthResponse(
            status="not_configured",
            configured=False,
            checked_at=datetime.now(UTC),
            components={
                "postgresql": component,
                "redis": component,
                "celery": component,
                "migrations": component,
            },
            metrics=InfrastructureMetrics(
                queue_length=None,
                pending_jobs=None,
                running_jobs=None,
                completed_jobs=None,
                failed_jobs=None,
                cache_hits=None,
                cache_misses=None,
                cache_keys=None,
                database_pool_size=None,
                database_checked_out_connections=None,
                celery_workers=None,
            ),
            migration_current=None,
            migration_head="0001_initial_schema",
        )

    async def shutdown(self) -> None:
        if self._cache is not None:
            await self._cache.close()
        if self._engine is not None:
            await dispose_engine(self._engine)
        self._cache = None
        self._engine = None
        self._service = None

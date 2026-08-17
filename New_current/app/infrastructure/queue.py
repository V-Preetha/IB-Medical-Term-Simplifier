"""Celery queue adapter and health inspection."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from celery import Celery, chain
from celery.exceptions import CeleryError
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.db.models import ProcessingStage
from app.infrastructure.errors import InfrastructureUnavailableError


@dataclass(frozen=True, slots=True)
class WorkerHealth:
    healthy: bool
    detail: str
    worker_count: int


class CeleryJobQueue:
    """Submit pipeline chains and expose worker/queue health."""

    def __init__(self, application: Celery, redis: Redis, *, health_timeout: int) -> None:
        self._application = application
        self._redis = redis
        self._health_timeout = health_timeout

    async def enqueue(
        self, job_id: UUID, request_id: UUID, stages: Sequence[ProcessingStage]
    ) -> str:
        signatures = [
            self._application.signature(
                f"ibhealth.pipeline.{stage.value}",
                kwargs={
                    "job_id": str(job_id),
                    "request_id": str(request_id),
                    "stage": stage.value,
                },
                immutable=True,
            )
            for stage in stages
        ]
        if not signatures:
            raise ValueError("At least one processing stage is required.")
        try:
            result = await asyncio.to_thread(chain(*signatures).apply_async)
        except (CeleryError, OSError) as exc:
            raise InfrastructureUnavailableError("Celery broker is unavailable.") from exc
        return str(result.id)

    async def revoke(self, celery_task_id: str) -> None:
        try:
            await asyncio.to_thread(
                self._application.control.revoke,
                celery_task_id,
                terminate=False,
            )
        except (CeleryError, OSError) as exc:
            raise InfrastructureUnavailableError("Celery control channel is unavailable.") from exc

    async def health(self) -> WorkerHealth:
        try:
            replies = await asyncio.to_thread(
                self._application.control.ping,
                timeout=self._health_timeout,
            )
        except (CeleryError, OSError):
            return WorkerHealth(False, "Celery workers are unreachable.", 0)
        worker_count = len(replies or [])
        return WorkerHealth(
            worker_count > 0,
            "Celery workers responded." if worker_count else "No Celery worker responded.",
            worker_count,
        )

    async def queue_length(self) -> int | None:
        try:
            queues = ("ibhealth.cpu", "ibhealth.gpu")
            lengths = await asyncio.gather(*(self._redis.llen(queue) for queue in queues))
        except RedisError:
            return None
        return sum(int(length) for length in lengths)

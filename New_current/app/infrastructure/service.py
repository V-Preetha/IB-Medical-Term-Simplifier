"""Transactional job lifecycle and infrastructure observability service."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from time import perf_counter
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.db.health import check_database_health
from app.db.models import ProcessingJob, ProcessingJobStatus, ProcessingStage, Report
from app.infrastructure.cache import RedisStageCache
from app.infrastructure.errors import (
    InfrastructureUnavailableError,
    JobConflictError,
    JobNotFoundError,
)
from app.infrastructure.queue import CeleryJobQueue
from app.infrastructure.schemas import (
    ComponentHealth,
    InfrastructureHealthResponse,
    InfrastructureMetrics,
    JobCreateRequest,
    JobResponse,
)

logger = logging.getLogger(__name__)
_MIGRATION_HEAD = "0001_initial_schema"


def _job_response(job: ProcessingJob, *, duplicate: bool = False) -> JobResponse:
    return JobResponse(
        job_id=job.job_id,
        request_id=job.request_id,
        report_id=job.report_id,
        stage=job.stage,
        requested_stages=[ProcessingStage(stage) for stage in job.requested_stages],
        stage_statuses=dict(job.stage_statuses),
        status=job.status,
        progress=job.progress_percent,
        attempt_count=job.attempt_count,
        pipeline_version=job.pipeline_version,
        configuration_version=job.configuration_version,
        model_revision=job.model_revision,
        celery_task_id=job.celery_task_id,
        error_code=job.error_code,
        error_message=job.error_message,
        queued_at=job.queued_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        created_at=job.created_at,
        updated_at=job.updated_at,
        version=job.version,
        duplicate=duplicate,
    )


class InfrastructureService:
    """Own job transactions, queue submission, cancellation, and health aggregation."""

    def __init__(
        self,
        *,
        engine: AsyncEngine,
        session_factory: async_sessionmaker[AsyncSession],
        cache: RedisStageCache,
        queue: CeleryJobQueue,
    ) -> None:
        self._engine = engine
        self._sessions = session_factory
        self._cache = cache
        self._queue = queue

    async def create_job(
        self, command: JobCreateRequest, *, owner_id: UUID, request_id: UUID
    ) -> JobResponse:
        stages = list(dict.fromkeys(command.stages))
        if stages != command.stages:
            raise JobConflictError("A processing stage may only appear once.")
        async with self._sessions() as session:
            report = (
                await session.execute(
                    select(Report).where(
                        Report.report_id == command.report_id,
                        Report.user_id == owner_id,
                        Report.deleted_at.is_(None),
                    )
                )
            ).scalar_one_or_none()
            if report is None:
                raise JobNotFoundError("The document was not found.")
            identity = {
                "configuration": command.configuration_version,
                "document": report.content_sha256,
                "model": command.model_revision,
                "owner": str(owner_id),
                "pipeline": command.pipeline_version,
                "stages": [stage.value for stage in stages],
            }
            idempotency_key = hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            existing = (
                await session.execute(
                    select(ProcessingJob).where(
                        ProcessingJob.idempotency_key == idempotency_key,
                        ProcessingJob.deleted_at.is_(None),
                    )
                )
            ).scalar_one_or_none()
            if existing is not None:
                return _job_response(existing, duplicate=True)
            now = datetime.now(UTC)
            job = ProcessingJob(
                request_id=request_id,
                report_id=report.report_id,
                stage=stages[0],
                pipeline_version=command.pipeline_version,
                configuration_version=command.configuration_version,
                model_revision=command.model_revision,
                idempotency_key=idempotency_key,
                requested_stages=[stage.value for stage in stages],
                stage_statuses={stage.value: "queued" for stage in stages},
                status=ProcessingJobStatus.PENDING,
                progress_percent=0,
                attempt_count=0,
                cancellation_requested=False,
                queued_at=now,
            )
            session.add(job)
            try:
                await session.commit()
                await session.refresh(job)
            except IntegrityError as exc:
                await session.rollback()
                raise JobConflictError("An equivalent processing job already exists.") from exc
            except SQLAlchemyError as exc:
                await session.rollback()
                raise InfrastructureUnavailableError(
                    "The processing job could not be stored."
                ) from exc
        try:
            task_id = await self._queue.enqueue(job.job_id, request_id, stages)
        except InfrastructureUnavailableError:
            await self._record_queue_failure(job.job_id)
            raise
        async with self._sessions() as session:
            persisted = await session.get(ProcessingJob, job.job_id)
            if persisted is None:
                raise InfrastructureUnavailableError("The acknowledged job could not be reloaded.")
            persisted.celery_task_id = task_id
            persisted.version += 1
            await session.commit()
            await session.refresh(persisted)
        logger.info(
            "Processing job queued",
            extra={
                "event": "processing_job_queued",
                "request_id": str(request_id),
                "job_id": str(job.job_id),
                "pipeline_stage": stages[0].value,
                "stage_count": len(stages),
            },
        )
        return _job_response(persisted)

    async def _record_queue_failure(self, job_id: UUID) -> None:
        async with self._sessions() as session:
            job = await session.get(ProcessingJob, job_id)
            if job is None:
                return
            job.error_code = "queue_unavailable"
            job.error_message = "The job is durable but has not been submitted to a worker."
            job.status = ProcessingJobStatus.RETRYING
            job.version += 1
            await session.commit()

    async def get_job(self, job_id: UUID, *, owner_id: UUID) -> JobResponse:
        async with self._sessions() as session:
            job = await self._owned_job(session, job_id, owner_id)
            if job is None:
                raise JobNotFoundError("The processing job was not found.")
            return _job_response(job)

    async def list_jobs(
        self,
        *,
        owner_id: UUID,
        status: ProcessingJobStatus | None,
        limit: int,
        offset: int,
    ) -> Sequence[JobResponse]:
        async with self._sessions() as session:
            statement = (
                select(ProcessingJob)
                .join(Report, ProcessingJob.report_id == Report.report_id)
                .where(Report.user_id == owner_id, ProcessingJob.deleted_at.is_(None))
                .order_by(ProcessingJob.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            if status is not None:
                statement = statement.where(ProcessingJob.status == status)
            jobs = (await session.execute(statement)).scalars().all()
            return [_job_response(job) for job in jobs]

    async def delete_job(self, job_id: UUID, *, owner_id: UUID) -> None:
        task_id: str | None
        async with self._sessions() as session:
            job = await self._owned_job(session, job_id, owner_id)
            if job is None:
                raise JobNotFoundError("The processing job was not found.")
            job.cancellation_requested = True
            job.status = ProcessingJobStatus.CANCELLED
            job.deleted_at = datetime.now(UTC)
            job.finished_at = datetime.now(UTC)
            job.version += 1
            task_id = job.celery_task_id
            await session.commit()
        if task_id:
            await self._queue.revoke(task_id)

    async def health(self) -> InfrastructureHealthResponse:
        checked_at = datetime.now(UTC)
        database_started = perf_counter()
        database = await check_database_health(self._engine)
        database_latency = (perf_counter() - database_started) * 1000
        redis_started = perf_counter()
        redis_healthy, redis_detail = await self._cache.health()
        redis_latency = (perf_counter() - redis_started) * 1000
        worker_started = perf_counter()
        worker = await self._queue.health()
        worker_latency = (perf_counter() - worker_started) * 1000
        migration_current = await self._migration_current()
        migrations_healthy = migration_current == _MIGRATION_HEAD
        stats = await self._cache.statistics()
        counts = await self._job_counts() if database.healthy else {}
        queue_length = await self._queue.queue_length() if redis_healthy else None
        components = {
            "postgresql": ComponentHealth(
                status="healthy" if database.healthy else "unhealthy",
                detail=database.detail,
                latency_ms=round(database_latency, 3),
            ),
            "redis": ComponentHealth(
                status="healthy" if redis_healthy else "unhealthy",
                detail=redis_detail,
                latency_ms=round(redis_latency, 3),
            ),
            "celery": ComponentHealth(
                status="healthy" if worker.healthy else "unhealthy",
                detail=worker.detail,
                latency_ms=round(worker_latency, 3),
            ),
            "migrations": ComponentHealth(
                status="healthy" if migrations_healthy else "unhealthy",
                detail=(
                    "Database schema is at the expected revision."
                    if migrations_healthy
                    else "Database schema is not at the expected revision."
                ),
            ),
        }
        overall = all(component.status == "healthy" for component in components.values())
        return InfrastructureHealthResponse(
            status="healthy" if overall else "unhealthy",
            configured=True,
            checked_at=checked_at,
            components=components,
            metrics=InfrastructureMetrics(
                queue_length=queue_length,
                pending_jobs=counts.get(ProcessingJobStatus.PENDING),
                running_jobs=counts.get(ProcessingJobStatus.RUNNING),
                completed_jobs=counts.get(ProcessingJobStatus.SUCCEEDED),
                failed_jobs=counts.get(ProcessingJobStatus.FAILED),
                cache_hits=stats["hits"],
                cache_misses=stats["misses"],
                cache_keys=stats["keys"],
                database_pool_size=database.pool_size,
                database_checked_out_connections=database.checked_out_connections,
                celery_workers=worker.worker_count,
            ),
            migration_current=migration_current,
            migration_head=_MIGRATION_HEAD,
        )

    async def _owned_job(
        self, session: AsyncSession, job_id: UUID, owner_id: UUID
    ) -> ProcessingJob | None:
        return (
            await session.execute(
                select(ProcessingJob)
                .join(Report, ProcessingJob.report_id == Report.report_id)
                .where(
                    ProcessingJob.job_id == job_id,
                    Report.user_id == owner_id,
                    ProcessingJob.deleted_at.is_(None),
                )
            )
        ).scalar_one_or_none()

    async def _migration_current(self) -> str | None:
        try:
            async with self._engine.connect() as connection:
                return (
                    await connection.execute(text("SELECT version_num FROM alembic_version"))
                ).scalar_one_or_none()
        except SQLAlchemyError:
            return None

    async def _job_counts(self) -> dict[ProcessingJobStatus, int]:
        try:
            async with self._sessions() as session:
                rows = (
                    await session.execute(
                        select(ProcessingJob.status, func.count(ProcessingJob.job_id))
                        .where(ProcessingJob.deleted_at.is_(None))
                        .group_by(ProcessingJob.status)
                    )
                ).all()
        except SQLAlchemyError:
            return {}
        return {status: int(count) for status, count in rows}

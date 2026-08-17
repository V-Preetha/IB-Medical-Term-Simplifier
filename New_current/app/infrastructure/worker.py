"""Worker-side stage-executor discovery and durable lifecycle updates."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from typing import Any, Protocol
from uuid import UUID

from redis.asyncio import Redis
from sqlalchemy import select

from app.db.config import DatabaseSettings
from app.db.models import ProcessingJob, ProcessingJobStatus, ProcessingStage, Report
from app.db.session import create_engine_and_sessionmaker, dispose_engine
from app.infrastructure.celery_app import create_celery_app
from app.infrastructure.config import InfrastructureSettings
from app.infrastructure.queue import CeleryJobQueue

logger = logging.getLogger(__name__)
_ENTRYPOINT_GROUP = "ib_health.pipeline_stages"


class TransientStageError(Exception):
    """Retryable dependency or capacity failure."""


class PermanentStageError(Exception):
    """Terminal invalid-input, configuration, or clinical-policy failure."""


@dataclass(frozen=True, slots=True)
class StageContext:
    job_id: UUID
    request_id: UUID
    report_id: UUID
    owner_id: UUID
    storage_url: str
    document_hash: str
    pipeline_version: str
    configuration_version: str
    model_revision: str


class StageExecutor(Protocol):
    """Infrastructure-neutral callable implemented by each AI stage adapter."""

    def __call__(self, context: StageContext) -> Awaitable[dict[str, Any]]: ...


def load_stage_executor(stage: ProcessingStage) -> StageExecutor:
    """Load exactly one deployment-provided executor for a pipeline stage."""

    matches = [
        entry
        for entry in metadata.entry_points(group=_ENTRYPOINT_GROUP)
        if entry.name == stage.value
    ]
    if len(matches) != 1:
        raise PermanentStageError(
            f"Stage '{stage.value}' requires exactly one registered executor; found {len(matches)}."
        )
    executor: Callable[[StageContext], Awaitable[dict[str, Any]]] = matches[0].load()
    return executor


async def execute_stage(job_id: UUID, stage: ProcessingStage) -> dict[str, Any]:
    """Run one registered stage and atomically persist safe lifecycle metadata."""

    engine, session_factory = create_engine_and_sessionmaker(DatabaseSettings.from_environment())
    try:
        async with session_factory() as session:
            result = await session.execute(
                select(ProcessingJob, Report)
                .join(Report, ProcessingJob.report_id == Report.report_id)
                .where(ProcessingJob.job_id == job_id, ProcessingJob.deleted_at.is_(None))
            )
            row = result.one_or_none()
            if row is None:
                raise PermanentStageError("The durable processing job does not exist.")
            job, report = row
            if job.cancellation_requested:
                job.status = ProcessingJobStatus.CANCELLED
                job.version += 1
                await session.commit()
                return {"status": "cancelled", "stage": stage.value}
            job.status = ProcessingJobStatus.RUNNING
            job.stage = stage
            job.attempt_count += 1
            job.started_at = job.started_at or datetime.now(UTC)
            job.stage_statuses = {**job.stage_statuses, stage.value: "running"}
            job.version += 1
            await session.commit()
            context = StageContext(
                job_id=job.job_id,
                request_id=job.request_id,
                report_id=report.report_id,
                owner_id=report.user_id,
                storage_url=report.storage_url,
                document_hash=report.content_sha256,
                pipeline_version=job.pipeline_version,
                configuration_version=job.configuration_version,
                model_revision=job.model_revision,
            )
        executor = load_stage_executor(stage)
        output = await executor(context)
        async with session_factory() as session:
            job = await session.get(ProcessingJob, job_id)
            if job is None:
                raise PermanentStageError("The durable processing job disappeared.")
            statuses = {**job.stage_statuses, stage.value: "completed"}
            completed = sum(value == "completed" for value in statuses.values())
            job.stage_statuses = statuses
            job.progress_percent = int(completed / len(job.requested_stages) * 100)
            job.status = (
                ProcessingJobStatus.SUCCEEDED
                if completed == len(job.requested_stages)
                else ProcessingJobStatus.RUNNING
            )
            if job.status == ProcessingJobStatus.SUCCEEDED:
                job.finished_at = datetime.now(UTC)
            job.version += 1
            await session.commit()
        logger.info(
            "Pipeline stage completed",
            extra={
                "event": "pipeline_stage_completed",
                "request_id": str(context.request_id),
                "job_id": str(job_id),
                "pipeline_stage": stage.value,
            },
        )
        return {"status": "completed", "stage": stage.value, "metadata": output}
    finally:
        await dispose_engine(engine)


async def mark_stage_failure(
    job_id: UUID, stage: ProcessingStage, *, retrying: bool, error_code: str
) -> None:
    """Persist a privacy-safe retry/failure state after task execution fails."""

    engine, session_factory = create_engine_and_sessionmaker(DatabaseSettings.from_environment())
    try:
        async with session_factory() as session:
            job = await session.get(ProcessingJob, job_id)
            if job is None:
                return
            state = "retrying" if retrying else "failed"
            job.status = ProcessingJobStatus.RETRYING if retrying else ProcessingJobStatus.FAILED
            job.stage_statuses = {**job.stage_statuses, stage.value: state}
            job.error_code = error_code
            job.error_message = "The processing stage did not complete."
            job.version += 1
            await session.commit()
    finally:
        await dispose_engine(engine)


async def recover_unsubmitted_jobs(limit: int = 100) -> int:
    """Resubmit durable jobs committed while the broker was unavailable."""

    settings = InfrastructureSettings.from_environment()
    engine, session_factory = create_engine_and_sessionmaker(DatabaseSettings.from_environment())
    redis = Redis.from_url(settings.redis_url, decode_responses=False)
    queue = CeleryJobQueue(
        create_celery_app(settings), redis, health_timeout=settings.health_timeout_seconds
    )
    recovered = 0
    try:
        async with session_factory() as session:
            jobs = (
                (
                    await session.execute(
                        select(ProcessingJob)
                        .where(
                            ProcessingJob.status == ProcessingJobStatus.RETRYING,
                            ProcessingJob.error_code == "queue_unavailable",
                            ProcessingJob.deleted_at.is_(None),
                        )
                        .order_by(ProcessingJob.queued_at.asc())
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )
            for job in jobs:
                stages = [ProcessingStage(value) for value in job.requested_stages]
                try:
                    task_id = await queue.enqueue(job.job_id, job.request_id, stages)
                except Exception:
                    logger.warning(
                        "Durable job recovery deferred",
                        extra={
                            "event": "processing_job_recovery_deferred",
                            "request_id": str(job.request_id),
                            "job_id": str(job.job_id),
                            "pipeline_stage": job.stage.value,
                        },
                    )
                    continue
                job.celery_task_id = task_id
                job.status = ProcessingJobStatus.PENDING
                job.error_code = None
                job.error_message = None
                job.version += 1
                recovered += 1
            await session.commit()
    finally:
        await redis.aclose()
        await dispose_engine(engine)
    return recovered

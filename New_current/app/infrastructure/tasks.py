"""Named Celery tasks for every requested processing stage."""

import asyncio
from uuid import UUID

from celery import Task

from app.db.models import ProcessingStage
from app.infrastructure.celery_app import celery_app
from app.infrastructure.config import InfrastructureSettings
from app.infrastructure.worker import (
    PermanentStageError,
    TransientStageError,
    execute_stage,
    mark_stage_failure,
    recover_unsubmitted_jobs,
)


def _execute(task: Task, *, job_id: str, stage: str) -> dict:
    settings = InfrastructureSettings.from_environment()
    job_uuid = UUID(job_id)
    pipeline_stage = ProcessingStage(stage)
    try:
        return asyncio.run(execute_stage(job_uuid, pipeline_stage))
    except TransientStageError as exc:
        retrying = task.request.retries < settings.celery_max_retries
        asyncio.run(
            mark_stage_failure(
                job_uuid,
                pipeline_stage,
                retrying=retrying,
                error_code="transient_stage_error",
            )
        )
        if not retrying:
            raise
        countdown = settings.celery_retry_backoff_seconds * (2**task.request.retries)
        raise task.retry(
            exc=exc,
            countdown=countdown,
            max_retries=settings.celery_max_retries,
        ) from exc
    except PermanentStageError:
        asyncio.run(
            mark_stage_failure(
                job_uuid,
                pipeline_stage,
                retrying=False,
                error_code="permanent_stage_error",
            )
        )
        raise
    except Exception:
        asyncio.run(
            mark_stage_failure(
                job_uuid,
                pipeline_stage,
                retrying=False,
                error_code="unexpected_stage_error",
            )
        )
        raise


def _register_stage_task(stage: ProcessingStage):
    @celery_app.task(bind=True, name=f"ibhealth.pipeline.{stage.value}")
    def stage_task(task: Task, *, job_id: str, request_id: str, stage: str) -> dict:
        del request_id
        return _execute(task, job_id=job_id, stage=stage)

    return stage_task


ocr_task = _register_stage_task(ProcessingStage.OCR)
ner_task = _register_stage_task(ProcessingStage.NER)
entity_linking_task = _register_stage_task(ProcessingStage.ENTITY_LINKING)
embeddings_task = _register_stage_task(ProcessingStage.EMBEDDINGS)
simplification_task = _register_stage_task(ProcessingStage.SIMPLIFICATION)
translation_task = _register_stage_task(ProcessingStage.TRANSLATION)


@celery_app.task(name="ibhealth.infrastructure.recover_jobs")
def recover_jobs_task() -> dict[str, int]:
    """Celery Beat entry point for broker-outage recovery."""

    return {"recovered": asyncio.run(recover_unsubmitted_jobs())}

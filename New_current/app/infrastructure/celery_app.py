"""Celery application configured for durable, idempotent pipeline work."""

from celery import Celery

from app.infrastructure.config import InfrastructureSettings


def create_celery_app(settings: InfrastructureSettings) -> Celery:
    """Create a Celery instance without importing FastAPI or model SDKs."""

    application = Celery(
        "ib_health_pipeline",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
        include=["app.infrastructure.tasks"],
    )
    application.conf.update(
        accept_content=["json"],
        task_serializer="json",
        result_serializer="json",
        enable_utc=True,
        timezone="UTC",
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_track_started=True,
        task_time_limit=settings.celery_task_timeout_seconds,
        task_soft_time_limit=max(1, settings.celery_task_timeout_seconds - 5),
        task_default_queue=settings.celery_default_queue,
        worker_prefetch_multiplier=1,
        result_expires=settings.cache_ttl_seconds,
        broker_connection_retry_on_startup=True,
        task_routes={
            "ibhealth.pipeline.ocr": {"queue": "ibhealth.gpu"},
            "ibhealth.pipeline.ner": {"queue": "ibhealth.cpu"},
            "ibhealth.pipeline.entity_linking": {"queue": "ibhealth.cpu"},
            "ibhealth.pipeline.embeddings": {"queue": "ibhealth.gpu"},
            "ibhealth.pipeline.simplification": {"queue": "ibhealth.gpu"},
            "ibhealth.pipeline.translation": {"queue": "ibhealth.gpu"},
            "ibhealth.infrastructure.recover_jobs": {"queue": "ibhealth.cpu"},
        },
        beat_schedule={
            "recover-durable-unsubmitted-jobs": {
                "task": "ibhealth.infrastructure.recover_jobs",
                "schedule": 60.0,
            }
        },
    )
    return application


celery_app = create_celery_app(InfrastructureSettings.from_environment())

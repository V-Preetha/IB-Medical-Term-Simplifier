"""Infrastructure API, health, dashboard, and OpenAPI tests."""

from datetime import UTC, datetime
from uuid import uuid4

from fastapi.testclient import TestClient

from app.db.models import ProcessingJobStatus, ProcessingStage
from app.infrastructure.config import InfrastructureSettings
from app.infrastructure.runtime import InfrastructureRuntime
from app.infrastructure.schemas import (
    ComponentHealth,
    InfrastructureHealthResponse,
    InfrastructureMetrics,
    JobResponse,
)
from app.main import create_app
from app.ocr.providers.runtime import ProviderContainer
from tests.ocr.fakes import FakeOCRProvider, FakePostProcessor


class _FakeJobService:
    def __init__(self) -> None:
        self.job_id = uuid4()

    def _response(self, report_id, request_id, *, duplicate=False) -> JobResponse:
        now = datetime.now(UTC)
        return JobResponse(
            job_id=self.job_id,
            request_id=request_id,
            report_id=report_id,
            stage=ProcessingStage.OCR,
            requested_stages=[ProcessingStage.OCR, ProcessingStage.NER],
            stage_statuses={"ocr": "queued", "ner": "queued"},
            status=ProcessingJobStatus.PENDING,
            progress=0,
            attempt_count=0,
            pipeline_version="pipeline-v1",
            configuration_version="config-v1",
            model_revision="model-revision-v1",
            celery_task_id="task-1",
            error_code=None,
            error_message=None,
            queued_at=now,
            started_at=None,
            finished_at=None,
            created_at=now,
            updated_at=now,
            version=1,
            duplicate=duplicate,
        )

    async def create_job(self, command, *, owner_id, request_id):
        del owner_id
        return self._response(command.report_id, request_id)

    async def get_job(self, job_id, *, owner_id):
        del job_id, owner_id
        return self._response(uuid4(), uuid4())

    async def list_jobs(self, *, owner_id, status, limit, offset):
        del owner_id, status, limit, offset
        return [self._response(uuid4(), uuid4())]

    async def delete_job(self, job_id, *, owner_id):
        del job_id, owner_id


class _FakeRuntime:
    def __init__(self) -> None:
        self.service = _FakeJobService()

    async def initialize(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def health(self) -> InfrastructureHealthResponse:
        component = ComponentHealth(status="healthy", detail="Synthetic dependency ready.")
        return InfrastructureHealthResponse(
            status="healthy",
            configured=True,
            checked_at=datetime.now(UTC),
            components={
                name: component for name in ("postgresql", "redis", "celery", "migrations")
            },
            metrics=InfrastructureMetrics(
                queue_length=0,
                pending_jobs=1,
                running_jobs=0,
                completed_jobs=2,
                failed_jobs=0,
                cache_hits=3,
                cache_misses=1,
                cache_keys=2,
                database_pool_size=5,
                database_checked_out_connections=1,
                celery_workers=1,
            ),
            migration_current="0001_initial_schema",
            migration_head="0001_initial_schema",
        )


def _application():
    return create_app(
        provider_container=ProviderContainer(FakeOCRProvider(), FakePostProcessor()),
        infrastructure_runtime=_FakeRuntime(),
    )


def test_job_crud_health_and_dashboard_contracts() -> None:
    report_id = uuid4()
    with TestClient(_application()) as client:
        created = client.post(
            "/api/v1/jobs",
            json={
                "report_id": str(report_id),
                "stages": ["ocr", "ner"],
                "pipeline_version": "pipeline-v1",
                "configuration_version": "config-v1",
                "model_revision": "model-revision-v1",
            },
        )
        fetched = client.get(f"/api/v1/jobs/{created.json()['job_id']}")
        listed = client.get("/api/v1/jobs")
        deleted = client.delete(f"/api/v1/jobs/{created.json()['job_id']}")
        health = client.get("/api/v1/infrastructure/health")
        dashboard = client.get("/infrastructure")
        javascript = client.get("/static/infrastructure_dashboard.js")
        openapi = client.get("/openapi.json").json()

    assert created.status_code == 202
    assert fetched.status_code == 200
    assert listed.status_code == 200
    assert deleted.status_code == 202
    assert health.status_code == 200
    assert health.json()["migration_current"] == "0001_initial_schema"
    assert dashboard.status_code == 200
    assert "Production Infrastructure" in dashboard.text
    assert javascript.status_code == 200
    for path in (
        "/api/v1/jobs",
        "/api/v1/jobs/{job_id}",
        "/api/v1/infrastructure/health",
    ):
        assert path in openapi["paths"]


def test_infrastructure_health_fails_closed_when_not_configured() -> None:
    settings = InfrastructureSettings(
        enabled=False,
        redis_url="redis://localhost:6379/0",
        redis_prefix="ibhealth",
        cache_ttl_seconds=3600,
        lock_ttl_seconds=300,
        cache_encryption_key="",
        celery_broker_url="redis://localhost:6379/1",
        celery_result_backend="redis://localhost:6379/2",
        celery_default_queue="ibhealth.cpu",
        celery_task_timeout_seconds=900,
        celery_max_retries=3,
        celery_retry_backoff_seconds=5,
        health_timeout_seconds=1,
    )
    application = create_app(
        provider_container=ProviderContainer(FakeOCRProvider(), FakePostProcessor()),
        infrastructure_runtime=InfrastructureRuntime(settings),
    )
    with TestClient(application) as client:
        response = client.get("/api/v1/infrastructure/health")
    assert response.status_code == 503
    assert response.json()["status"] == "not_configured"
    assert response.json()["configured"] is False

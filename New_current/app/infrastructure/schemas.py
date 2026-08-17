"""Versioned job and infrastructure API schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import ProcessingJobStatus, ProcessingStage


class JobCreateRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "report_id": "11111111-1111-4111-8111-111111111111",
                    "stages": ["ocr", "ner"],
                    "pipeline_version": "clinical-pipeline-v1",
                    "configuration_version": "deployment-2026-08",
                    "model_revision": "immutable-revision-set-v1",
                }
            ]
        },
    )

    report_id: UUID
    stages: list[ProcessingStage] = Field(min_length=1, max_length=6)
    pipeline_version: str = Field(min_length=1, max_length=64)
    configuration_version: str = Field(min_length=1, max_length=128)
    model_revision: str = Field(min_length=1, max_length=255)


class JobResponse(BaseModel):
    schema_version: str = "processing-job-v1"
    job_id: UUID
    request_id: UUID
    report_id: UUID
    stage: ProcessingStage
    requested_stages: list[ProcessingStage]
    stage_statuses: dict[str, str]
    status: ProcessingJobStatus
    progress: int
    attempt_count: int
    pipeline_version: str
    configuration_version: str
    model_revision: str
    celery_task_id: str | None
    error_code: str | None
    error_message: str | None
    queued_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime
    version: int
    duplicate: bool = False


class JobListResponse(BaseModel):
    schema_version: str = "processing-job-list-v1"
    items: list[JobResponse]
    count: int


class JobDeleteResponse(BaseModel):
    schema_version: str = "processing-job-delete-v1"
    job_id: UUID
    status: str


class ComponentHealth(BaseModel):
    status: str
    detail: str
    latency_ms: float | None = None


class InfrastructureMetrics(BaseModel):
    queue_length: int | None
    pending_jobs: int | None
    running_jobs: int | None
    completed_jobs: int | None
    failed_jobs: int | None
    cache_hits: int | None
    cache_misses: int | None
    cache_keys: int | None
    database_pool_size: int | None
    database_checked_out_connections: int | None
    celery_workers: int | None


class InfrastructureHealthResponse(BaseModel):
    schema_version: str = "infrastructure-health-v1"
    status: str
    configured: bool
    checked_at: datetime
    components: dict[str, ComponentHealth]
    metrics: InfrastructureMetrics
    migration_current: str | None
    migration_head: str


class ErrorDetail(BaseModel):
    code: str
    message: str


class InfrastructureErrorResponse(BaseModel):
    error: ErrorDetail
    request_id: UUID


class GPUMetrics(BaseModel):
    """Process-wide CUDA memory metrics. ``None`` means not measurable here."""

    available: bool
    device_name: str | None
    allocated_mb: float | None
    reserved_mb: float | None
    peak_allocated_mb: float | None
    total_mb: float | None
    utilization_percent: float | None
    utilization_source: str | None = None


class CPUMetrics(BaseModel):
    """Metrics for this API process only, not the whole host."""

    process_rss_mb: float | None
    process_cpu_percent: float | None


class ModelRuntimeStatus(BaseModel):
    """Cold/warm and identity status for one active provider."""

    stage: str
    provider_name: str | None
    model_name: str | None
    model_revision: str | None
    device: str | None
    loaded: bool
    warm: bool
    load_timestamp: datetime | None
    load_duration_ms: float | None
    request_count: int | None


class RuntimeMetricsResponse(BaseModel):
    schema_version: str = "runtime-metrics-v1"
    checked_at: datetime
    gpu: GPUMetrics
    cpu: CPUMetrics
    models: list[ModelRuntimeStatus]

"""Versioned job and production-infrastructure HTTP endpoints."""

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import psutil
from fastapi import APIRouter, Header, Query, Request, Response, status

from app.config import settings as application_settings
from app.db.models import ProcessingJobStatus
from app.infrastructure.runtime import InfrastructureRuntime
from app.infrastructure.schemas import (
    CPUMetrics,
    GPUMetrics,
    InfrastructureErrorResponse,
    InfrastructureHealthResponse,
    JobCreateRequest,
    JobDeleteResponse,
    JobListResponse,
    JobResponse,
    ModelRuntimeStatus,
    RuntimeMetricsResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Production Infrastructure"])
_OWNER_HEADER = Header(default=None, alias="X-Owner-ID")
_JOB_STATUS_QUERY = Query(default=None, alias="status")
_LIMIT_QUERY = Query(default=50, ge=1, le=200)
_OFFSET_QUERY = Query(default=0, ge=0)
_BYTES_PER_MB = 1024 * 1024
# A psutil.Process handle must be reused across calls: cpu_percent() reports
# usage since the *previous* call on the same handle, so a fresh handle per
# request would always report 0.0. Priming it once at import time discards
# the meaningless zero-length first interval.
_PROCESS_HANDLE = psutil.Process()
_PROCESS_HANDLE.cpu_percent(interval=None)


def _runtime(request: Request) -> InfrastructureRuntime:
    return request.app.state.infrastructure_runtime


def _owner_id(value: UUID | None) -> UUID:
    return value or application_settings.default_owner_id


@router.post(
    "/api/v1/jobs",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create a durable processing job",
    description=(
        "Creates an idempotent tenant-scoped pipeline job, commits it to PostgreSQL, "
        "and submits its ordered stages to Celery. Equivalent active requests return "
        "the existing job without duplicate execution."
    ),
    responses={
        202: {"description": "The durable job was queued."},
        404: {
            "model": InfrastructureErrorResponse,
            "description": "The tenant-owned document does not exist.",
        },
        409: {
            "model": InfrastructureErrorResponse,
            "description": "The requested stage sequence is invalid or conflicts.",
        },
        503: {
            "model": InfrastructureErrorResponse,
            "description": "PostgreSQL, Redis, or Celery is unavailable.",
        },
    },
)
async def create_job(
    command: JobCreateRequest,
    request: Request,
    x_owner_id: UUID | None = _OWNER_HEADER,
) -> JobResponse:
    return await _runtime(request).service.create_job(
        command,
        owner_id=_owner_id(x_owner_id),
        request_id=request.state.request_id,
    )


@router.get(
    "/api/v1/jobs/{job_id}",
    response_model=JobResponse,
    summary="Get processing job status",
    description="Returns tenant-scoped durable status, stage progress, retries, and errors.",
    responses={
        404: {
            "model": InfrastructureErrorResponse,
            "description": "The processing job does not exist.",
        }
    },
)
async def get_job(
    job_id: UUID,
    request: Request,
    x_owner_id: UUID | None = _OWNER_HEADER,
) -> JobResponse:
    return await _runtime(request).service.get_job(job_id, owner_id=_owner_id(x_owner_id))


@router.get(
    "/api/v1/jobs",
    response_model=JobListResponse,
    summary="List processing jobs",
    description="Lists recent tenant-owned jobs with optional lifecycle filtering.",
)
async def list_jobs(
    request: Request,
    x_owner_id: UUID | None = _OWNER_HEADER,
    job_status: ProcessingJobStatus | None = _JOB_STATUS_QUERY,
    limit: int = _LIMIT_QUERY,
    offset: int = _OFFSET_QUERY,
) -> JobListResponse:
    items = await _runtime(request).service.list_jobs(
        owner_id=_owner_id(x_owner_id),
        status=job_status,
        limit=limit,
        offset=offset,
    )
    return JobListResponse(items=list(items), count=len(items))


@router.delete(
    "/api/v1/jobs/{job_id}",
    response_model=JobDeleteResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Cancel and delete a processing job",
    description=(
        "Marks the tenant-owned job cancelled and soft-deleted, then sends a safe "
        "non-terminating revoke request to Celery."
    ),
    responses={
        404: {
            "model": InfrastructureErrorResponse,
            "description": "The processing job does not exist.",
        }
    },
)
async def delete_job(
    job_id: UUID,
    request: Request,
    x_owner_id: UUID | None = _OWNER_HEADER,
) -> JobDeleteResponse:
    await _runtime(request).service.delete_job(job_id, owner_id=_owner_id(x_owner_id))
    return JobDeleteResponse(job_id=job_id, status="cancelled")


@router.get(
    "/api/v1/infrastructure/health",
    response_model=InfrastructureHealthResponse,
    summary="Inspect production infrastructure health",
    description=(
        "Checks PostgreSQL connectivity and pool metrics, Redis, Celery workers and "
        "queues, Alembic revision state, cache statistics, and durable job counts."
    ),
    responses={
        200: {"description": "Every required infrastructure component is healthy."},
        503: {"description": "Returns typed health with an unconfigured or unhealthy status."},
    },
)
async def infrastructure_health(
    request: Request, response: Response
) -> InfrastructureHealthResponse:
    health = await _runtime(request).health()
    if health.status != "healthy":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return health


@router.get(
    "/api/v1/runtime/metrics",
    response_model=RuntimeMetricsResponse,
    summary="Inspect internal runtime performance metrics",
    description=(
        "Returns safe, non-clinical process/GPU/CPU metrics and per-stage model "
        "load/warm status for engineering performance debugging. Never returns "
        "secrets, environment variable contents, or clinical text. A metric is "
        "`null` when this host/runtime cannot measure it reliably; no value is "
        "invented as a substitute."
    ),
)
async def runtime_metrics(request: Request) -> RuntimeMetricsResponse:
    return RuntimeMetricsResponse(
        checked_at=datetime.now(UTC),
        gpu=_gpu_metrics(),
        cpu=_cpu_metrics(),
        models=_model_statuses(request),
    )


def _gpu_metrics() -> GPUMetrics:
    empty = GPUMetrics(
        available=False,
        device_name=None,
        allocated_mb=None,
        reserved_mb=None,
        peak_allocated_mb=None,
        total_mb=None,
        utilization_percent=None,
    )
    try:
        import torch
    except ImportError:
        return empty
    if not torch.cuda.is_available():
        return empty
    device = torch.cuda.current_device()
    utilization: float | None = None
    utilization_source: str | None = None
    try:
        utilization = float(torch.cuda.utilization(device))
        utilization_source = "torch.cuda.utilization"
    except Exception:  # noqa: BLE001 - utilization is best-effort observability only
        utilization = None
        utilization_source = None
    return GPUMetrics(
        available=True,
        device_name=torch.cuda.get_device_name(device),
        allocated_mb=round(torch.cuda.memory_allocated(device) / _BYTES_PER_MB, 3),
        reserved_mb=round(torch.cuda.memory_reserved(device) / _BYTES_PER_MB, 3),
        peak_allocated_mb=round(torch.cuda.max_memory_allocated(device) / _BYTES_PER_MB, 3),
        total_mb=round(
            torch.cuda.get_device_properties(device).total_memory / _BYTES_PER_MB, 3
        ),
        utilization_percent=utilization,
        utilization_source=utilization_source,
    )


def _cpu_metrics() -> CPUMetrics:
    try:
        rss_mb = round(_PROCESS_HANDLE.memory_info().rss / _BYTES_PER_MB, 3)
    except Exception:  # noqa: BLE001 - process metrics are best-effort observability only
        logger.warning("Process RSS could not be measured", exc_info=True)
        rss_mb = None
    try:
        cpu_percent = _PROCESS_HANDLE.cpu_percent(interval=None)
    except Exception:  # noqa: BLE001 - process metrics are best-effort observability only
        logger.warning("Process CPU utilization could not be measured", exc_info=True)
        cpu_percent = None
    return CPUMetrics(process_rss_mb=rss_mb, process_cpu_percent=cpu_percent)


def _as_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _status_from_metadata(stage: str, metadata: Any) -> ModelRuntimeStatus:
    """Build a ``ModelRuntimeStatus`` from any provider metadata object.

    Every provider's metadata object exposes ``configuration`` (a free-form
    dict) and sets ``startup_timestamp``/``model_loading_time_ms`` only after
    a real successful load, so those two facts are read generically here
    without importing any single stage's concrete metadata type.
    """

    configuration = dict(getattr(metadata, "configuration", None) or {})
    load_timestamp = _as_timestamp(getattr(metadata, "startup_timestamp", None))
    if load_timestamp is None:
        load_timestamp = _as_timestamp(configuration.get("startup_timestamp"))
    loaded = load_timestamp is not None
    load_duration = getattr(metadata, "loading_time_ms", None)
    if load_duration is None:
        load_duration = getattr(metadata, "model_loading_time_ms", None)
    if load_duration is None:
        load_duration = configuration.get("model_loading_time_ms")
    request_count = configuration.get("request_count")
    return ModelRuntimeStatus(
        stage=stage,
        provider_name=getattr(metadata, "provider_name", None),
        model_name=getattr(metadata, "model_name", None) or configuration.get("model_name"),
        model_revision=(
            getattr(metadata, "model_revision", None) or configuration.get("model_revision")
        ),
        device=(
            getattr(metadata, "device", None)
            or configuration.get("resolved_device")
            or configuration.get("device")
        ),
        loaded=loaded,
        warm=bool(configuration.get("warm", loaded)),
        load_timestamp=load_timestamp,
        load_duration_ms=load_duration,
        request_count=request_count if isinstance(request_count, int) else None,
    )


def _model_statuses(request: Request) -> list[ModelRuntimeStatus]:
    state = request.app.state
    statuses: list[ModelRuntimeStatus] = []

    ocr_service = getattr(state, "ocr_application_service", None)
    if ocr_service is not None:
        try:
            ocr_metadata, postprocessor_metadata = ocr_service.provider_metadata()
            statuses.append(_status_from_metadata("ocr", ocr_metadata))
            statuses.append(_status_from_metadata("ocr_postprocessing", postprocessor_metadata))
        except Exception:  # noqa: BLE001 - one stage's failure must not break the endpoint
            logger.warning("OCR runtime status could not be read", exc_info=True)

    for stage, attribute in (
        ("ner", "ner_service"),
        ("embeddings", "embedding_service"),
    ):
        service = getattr(state, attribute, None)
        if service is None:
            continue
        try:
            statuses.append(_status_from_metadata(stage, service.health().metadata))
        except Exception:  # noqa: BLE001 - one stage's failure must not break the endpoint
            logger.warning("%s runtime status could not be read", stage, exc_info=True)

    for stage, attribute in (
        ("simplification", "simplification_service"),
        ("translation", "translation_service"),
        ("verification", "verification_service"),
    ):
        service = getattr(state, attribute, None)
        if service is None:
            continue
        try:
            statuses.append(_status_from_metadata(stage, service.provider.metadata()))
        except Exception:  # noqa: BLE001 - one stage's failure must not break the endpoint
            logger.warning("%s runtime status could not be read", stage, exc_info=True)

    return statuses

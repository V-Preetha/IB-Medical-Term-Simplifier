"""Versioned HTTP routes for the single OCR application service."""

from pathlib import Path
from time import perf_counter
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, Query, Request, Response, UploadFile, status

from app.config import settings
from app.ocr.api.dependencies import (
    LogStoreDependency,
    OCRApplicationServiceDependency,
    OwnerDependency,
)
from app.ocr.api.schemas import (
    ErrorDetail,
    ErrorResponse,
    OCRLogResponse,
    OCRModelResponse,
    OCRModelsResponse,
    OCRRecentResponse,
    OCRResponse,
    OCRStatusResponse,
    ProviderHealthResponse,
    ProviderStatusResponse,
    ServiceHealthResponse,
)
from app.ocr.application.errors import (
    OCRNotFoundError,
    OCRUploadError,
    OCRUploadTooLargeError,
)
from app.ocr.domain.records import OCRRequestRecord, OCRRequestStatus, OCRResultRecord
from app.ocr.providers.contracts import ProviderHealthStatus
from app.ocr.providers.errors import UnsupportedDocumentError

router = APIRouter(prefix="/api/v1/ocr", tags=["OCR"])
service_health_router = APIRouter(prefix="/api/v1/health", tags=["Health"])

_FILE_TYPES = {
    ".pdf": "pdf",
    ".png": "png",
    ".jpg": "jpg",
    ".jpeg": "jpeg",
    ".tif": "tiff",
    ".tiff": "tiff",
    ".bmp": "bmp",
    ".webp": "webp",
    ".heic": "heic",
    ".heif": "heif",
}

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "The upload or request is invalid."},
    404: {"model": ErrorResponse, "description": "The OCR request was not found."},
    413: {"model": ErrorResponse, "description": "The upload exceeds the configured limit."},
    415: {"model": ErrorResponse, "description": "The document type is unsupported."},
    500: {"model": ErrorResponse, "description": "OCR inference failed safely."},
    503: {"model": ErrorResponse, "description": "A configured OCR provider is unavailable."},
}


@router.post(
    "",
    response_model=OCRResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Run the OCR pipeline",
    description=(
        "Validates one de-identified document and executes configured document "
        "Qwen3-VL OCR and medical post-processing providers."
    ),
    responses=_ERROR_RESPONSES,
)
async def create_ocr(
    request: Request,
    service: OCRApplicationServiceDependency,
    owner_id: OwnerDependency,
    file: Annotated[
        UploadFile,
        File(
            description=(
                "PDF, PNG, JPEG, TIFF, BMP, WebP, or HEIC clinical document. "
                "Use synthetic or approved de-identified inputs in the engineering console."
            )
        ),
    ],
) -> OCRResponse:
    filename = Path(file.filename or "").name
    if not filename:
        raise OCRUploadError("A filename is required.")
    file_type = _FILE_TYPES.get(Path(filename).suffix.casefold())
    if file_type is None:
        raise UnsupportedDocumentError("The uploaded file extension is unsupported.")
    upload_started = perf_counter()
    content = await file.read(settings.max_upload_bytes + 1)
    await file.close()
    upload_time_ms = round((perf_counter() - upload_started) * 1000, 3)
    if not content:
        raise OCRUploadError("The uploaded document is empty.")
    if len(content) > settings.max_upload_bytes:
        raise OCRUploadTooLargeError("The uploaded document exceeds the configured size limit.")
    result = await service.process(
        request_id=request.state.request_id,
        owner_id=owner_id,
        content=content,
        filename=filename,
        file_type=file_type,
        media_type=file.content_type or "application/octet-stream",
        upload_time_ms=upload_time_ms,
    )
    return _result_response(result)


@router.get(
    "/status/{request_id}",
    response_model=OCRStatusResponse,
    summary="Get OCR processing status",
    description="Returns the lifecycle state and progress for one tenant-owned OCR request.",
    responses={404: _ERROR_RESPONSES[404]},
)
async def get_ocr_status(
    request_id: UUID,
    service: OCRApplicationServiceDependency,
    owner_id: OwnerDependency,
) -> OCRStatusResponse:
    record = await service.get_status(request_id, owner_id)
    if record is None:
        raise OCRNotFoundError("The OCR request was not found.")
    return _status_response(record)


@router.get(
    "/recent",
    response_model=OCRRecentResponse,
    summary="List recent OCR requests",
    description="Returns bounded process-local request history for the engineering console.",
)
async def recent_ocr_requests(
    service: OCRApplicationServiceDependency,
    owner_id: OwnerDependency,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> OCRRecentResponse:
    records = await service.list_recent(owner_id, limit=limit)
    return OCRRecentResponse(requests=[_status_response(record) for record in records])


@router.get(
    "/logs",
    response_model=OCRLogResponse,
    summary="Inspect recent OCR logs",
    description=(
        "Returns bounded privacy-safe structured events for the internal engineering console."
    ),
)
async def recent_ocr_logs(
    store: LogStoreDependency,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
) -> OCRLogResponse:
    return OCRLogResponse(records=[dict(record) for record in store.recent(limit)])


@router.get(
    "/models",
    response_model=OCRModelsResponse,
    summary="List configured OCR models",
    description="Returns configuration-safe metadata for every provider in the OCR pipeline.",
    responses={503: _ERROR_RESPONSES[503]},
)
async def get_ocr_models(service: OCRApplicationServiceDependency) -> OCRModelsResponse:
    models = []
    for metadata in service.provider_metadata():
        configuration = dict(metadata.configuration)
        models.append(
            OCRModelResponse(
                provider_name=metadata.provider_name,
                provider_version=metadata.provider_version,
                provider_kind=metadata.provider_kind.value,
                model_name=_optional_string(configuration.get("model_name")),
                model_revision=_optional_string(configuration.get("model_revision")),
                supported_file_types=list(metadata.supported_file_types),
                supported_document_types=list(metadata.supported_document_types),
                configuration=configuration,
            )
        )
    return OCRModelsResponse(models=models)


@router.get(
    "/health",
    response_model=ProviderHealthResponse,
    summary="Inspect OCR provider health",
    description=(
        "Returns lifecycle readiness and configuration-safe capability metadata for every "
        "configured provider without executing inference."
    ),
    responses={503: _ERROR_RESPONSES[503]},
)
async def get_ocr_health(service: OCRApplicationServiceDependency) -> ProviderHealthResponse:
    providers = []
    states = []
    for metadata, health in zip(
        service.provider_metadata(), service.provider_health(), strict=True
    ):
        states.append(health.status)
        providers.append(
            ProviderStatusResponse(
                provider_name=metadata.provider_name,
                provider_version=metadata.provider_version,
                provider_kind=metadata.provider_kind.value,
                health_status=health.status,
                health_detail=health.detail,
                startup_timestamp=health.startup_timestamp,
                checked_at=health.checked_at,
                supported_file_types=list(metadata.supported_file_types),
                supported_document_types=list(metadata.supported_document_types),
                configuration=dict(metadata.configuration),
            )
        )
    return ProviderHealthResponse(status=_overall_status(states), providers=providers)


@router.get(
    "/{request_id}",
    response_model=OCRResponse,
    summary="Get an OCR result",
    description="Returns one completed tenant-owned OCR result and its provenance metadata.",
    responses={404: _ERROR_RESPONSES[404]},
)
async def get_ocr_result(
    request_id: UUID,
    service: OCRApplicationServiceDependency,
    owner_id: OwnerDependency,
) -> OCRResponse:
    result = await service.get_result(request_id, owner_id)
    if result is None:
        raise OCRNotFoundError("The OCR result was not found.")
    return _result_response(result)


@router.delete(
    "/{request_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an OCR result",
    description="Deletes one tenant-owned process-local OCR request, result, and cache entry.",
    responses={404: _ERROR_RESPONSES[404]},
)
async def delete_ocr_result(
    request_id: UUID,
    service: OCRApplicationServiceDependency,
    owner_id: OwnerDependency,
) -> Response:
    if not await service.delete(request_id, owner_id):
        raise OCRNotFoundError("The OCR request was not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@service_health_router.get(
    "/live",
    response_model=ServiceHealthResponse,
    summary="Check OCR API liveness",
    description="Reports whether the FastAPI process can serve requests.",
)
async def liveness() -> ServiceHealthResponse:
    return ServiceHealthResponse(status="live", service="ocr")


@service_health_router.get(
    "/ready",
    response_model=ServiceHealthResponse,
    summary="Check OCR API readiness",
    description="Reports readiness only when the configured OCR provider container is ready.",
    responses={503: _ERROR_RESPONSES[503]},
)
async def readiness(
    service: OCRApplicationServiceDependency, response: Response
) -> ServiceHealthResponse:
    if any(health.status is not ProviderHealthStatus.READY for health in service.provider_health()):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ServiceHealthResponse(status="not_ready", service="ocr")
    return ServiceHealthResponse(status="ready", service="ocr")


def _result_response(result: OCRResultRecord) -> OCRResponse:
    return OCRResponse(
        request_id=result.request_id,
        report_id=result.report_id,
        ocr_id=result.process_id,
        document_type=result.document_type,
        provider=result.provider_name,
        provider_version=result.provider_version,
        pipeline_version=result.pipeline_version,
        confidence=result.confidence,
        confidence_method=result.confidence_method,
        processing_time_ms=result.processing_time_ms,
        page_count=result.page_count,
        status=OCRRequestStatus.COMPLETED,
        raw_text=result.raw_text,
        normalized_text=result.normalized_text,
        cache_hit=result.cache_hit,
        review_required=result.review_state.value == "required",
        warnings=list(result.warnings),
        metadata={
            **dict(result.metadata),
            "model_name": result.model_name,
            "model_version": result.model_version,
            "rule_version": result.rule_version,
            "schema_version": result.schema_version,
            "calibration_version": result.calibration_version,
        },
        created_at=result.created_at,
    )


def _status_response(record: OCRRequestRecord) -> OCRStatusResponse:
    error = (
        ErrorDetail(code=record.error_code, message=record.error_message or "OCR failed.")
        if record.error_code
        else None
    )
    return OCRStatusResponse(
        request_id=record.request_id,
        report_id=record.report_id,
        ocr_id=record.process_id,
        status=record.status,
        pipeline_stage=record.stage,
        progress=record.progress_percent,
        error=error,
        created_at=record.created_at,
        updated_at=record.updated_at,
        completed_at=record.completed_at,
    )


def _overall_status(states: list[ProviderHealthStatus]) -> ProviderHealthStatus:
    if all(state is ProviderHealthStatus.READY for state in states):
        return ProviderHealthStatus.READY
    if any(state is ProviderHealthStatus.UNAVAILABLE for state in states):
        return ProviderHealthStatus.UNAVAILABLE
    if any(state is ProviderHealthStatus.DEGRADED for state in states):
        return ProviderHealthStatus.DEGRADED
    if all(state is ProviderHealthStatus.STOPPED for state in states):
        return ProviderHealthStatus.STOPPED
    return ProviderHealthStatus.INITIALIZING


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None

"""Public Pydantic contracts for the versioned OCR API."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.ocr.domain.records import OCRRequestStatus
from app.ocr.providers.contracts import ProviderHealthStatus


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
    request_id: str | None = None


class OCRResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_id": "8be19f66-b495-4f98-ab89-acde20065f2f",
                "report_id": "7ef8df31-122c-4e77-af37-82076f70a020",
                "ocr_id": "f16dad72-1bf5-4314-b9a4-adc63660bd26",
                "document_type": "scanned_pdf",
                "provider": "qwen3-vl",
                "provider_version": "1.0.0",
                "pipeline_version": "phase2-ocr-v1",
                "confidence": 0.94,
                "confidence_method": "mean_generated_token_probability",
                "processing_time_ms": 1840.2,
                "page_count": 2,
                "status": "completed",
                "raw_text": "Synthetic de-identified report text.",
                "normalized_text": "Synthetic de-identified report text.",
                "cache_hit": False,
                "review_required": False,
                "warnings": [],
                "metadata": {"schema_version": "ocr-response-v1"},
                "created_at": "2026-08-03T10:00:00Z",
            }
        }
    )

    request_id: UUID
    report_id: UUID
    ocr_id: UUID
    document_type: str
    provider: str
    provider_version: str
    pipeline_version: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_method: str | None
    processing_time_ms: float = Field(ge=0)
    page_count: int = Field(ge=1)
    status: OCRRequestStatus
    raw_text: str
    normalized_text: str
    cache_hit: bool
    review_required: bool
    warnings: list[str]
    metadata: dict[str, Any]
    created_at: datetime


class OCRStatusResponse(BaseModel):
    request_id: UUID
    report_id: UUID
    ocr_id: UUID
    status: OCRRequestStatus
    pipeline_stage: str
    progress: int = Field(ge=0, le=100)
    error: ErrorDetail | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


class ProviderStatusResponse(BaseModel):
    provider_name: str
    provider_version: str
    provider_kind: str
    health_status: ProviderHealthStatus
    health_detail: str
    startup_timestamp: datetime | None
    checked_at: datetime
    supported_file_types: list[str]
    supported_document_types: list[str]
    configuration: dict[str, Any]


class ProviderHealthResponse(BaseModel):
    status: ProviderHealthStatus
    providers: list[ProviderStatusResponse]


class OCRModelResponse(BaseModel):
    provider_name: str
    provider_version: str
    provider_kind: str
    model_name: str | None
    model_revision: str | None
    supported_file_types: list[str]
    supported_document_types: list[str]
    configuration: dict[str, Any]


class OCRModelsResponse(BaseModel):
    models: list[OCRModelResponse]


class OCRRecentResponse(BaseModel):
    requests: list[OCRStatusResponse]


class OCRLogResponse(BaseModel):
    records: list[dict[str, Any]]


class ServiceHealthResponse(BaseModel):
    status: str
    service: str

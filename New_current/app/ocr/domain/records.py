"""Persistence-neutral records for OCR processing state and results."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from re import fullmatch
from types import MappingProxyType
from typing import Any
from uuid import UUID


class OCRRequestStatus(str, Enum):
    """Durable lifecycle states exposed by the OCR application boundary."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class OCRReviewState(str, Enum):
    """Clinical review disposition kept separate from execution state."""

    NOT_REQUIRED = "not_required"
    REQUIRED = "required"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class OCRRequestRecord:
    """Persistence-neutral snapshot of one tenant-owned OCR request."""

    request_id: UUID
    report_id: UUID
    process_id: UUID
    owner_id: UUID
    content_sha256: str
    original_filename: str
    media_type: str
    status: OCRRequestStatus
    stage: str
    progress_percent: int
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if fullmatch(r"[0-9a-f]{64}", self.content_sha256) is None:
            raise ValueError("content_sha256 must be a lowercase SHA-256 digest")
        if not self.original_filename.strip():
            raise ValueError("original_filename must not be blank")
        if not self.media_type.strip():
            raise ValueError("media_type must not be blank")
        if not self.stage.strip():
            raise ValueError("stage must not be blank")
        if not 0 <= self.progress_percent <= 100:
            raise ValueError("progress_percent must be between 0 and 100")
        for field_name in ("created_at", "updated_at", "completed_at"):
            value = getattr(self, field_name)
            if value is not None and value.tzinfo is None:
                raise ValueError(f"{field_name} must be timezone-aware")
        if self.status is OCRRequestStatus.COMPLETED and self.completed_at is None:
            raise ValueError("completed requests require completed_at")
        if self.status is OCRRequestStatus.FAILED and not self.error_code:
            raise ValueError("failed requests require an error_code")
        if self.status is not OCRRequestStatus.FAILED and (
            self.error_code is not None or self.error_message is not None
        ):
            raise ValueError("only failed requests may contain error details")

    @classmethod
    def queued(
        cls,
        *,
        request_id: UUID,
        report_id: UUID,
        process_id: UUID,
        owner_id: UUID,
        content_sha256: str,
        original_filename: str,
        media_type: str,
        created_at: datetime | None = None,
    ) -> "OCRRequestRecord":
        """Create a validated initial record without depending on a clock service."""

        timestamp = created_at or datetime.now(UTC)
        return cls(
            request_id=request_id,
            report_id=report_id,
            process_id=process_id,
            owner_id=owner_id,
            content_sha256=content_sha256,
            original_filename=original_filename,
            media_type=media_type,
            status=OCRRequestStatus.QUEUED,
            stage="queued",
            progress_percent=0,
            created_at=timestamp,
            updated_at=timestamp,
        )


@dataclass(frozen=True, slots=True)
class OCRResultRecord:
    """Reproducible OCR output supplied to a future persistence adapter."""

    request_id: UUID
    report_id: UUID
    process_id: UUID
    raw_text: str
    normalized_text: str
    document_type: str
    page_count: int
    provider_name: str
    model_name: str
    model_version: str
    provider_version: str
    pipeline_version: str
    rule_version: str
    schema_version: str
    confidence: float | None
    confidence_method: str | None
    calibration_version: str | None
    processing_time_ms: float
    cache_hit: bool
    review_state: OCRReviewState
    warnings: tuple[str, ...]
    metadata: Mapping[str, Any]
    created_at: datetime

    def __post_init__(self) -> None:
        required_strings = (
            "document_type",
            "provider_name",
            "model_name",
            "model_version",
            "provider_version",
            "pipeline_version",
            "rule_version",
            "schema_version",
        )
        if any(not getattr(self, name).strip() for name in required_strings):
            raise ValueError("OCR result version and model fields must not be blank")
        if self.page_count <= 0:
            raise ValueError("page_count must be positive")
        if self.processing_time_ms < 0:
            raise ValueError("processing_time_ms must not be negative")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.confidence is None and self.confidence_method is not None:
            raise ValueError("confidence_method requires confidence")
        if self.confidence is not None and not self.confidence_method:
            raise ValueError("confidence requires confidence_method")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

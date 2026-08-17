"""Environment-backed configuration for the converged OCR service."""

import os
from dataclasses import dataclass
from uuid import UUID


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Process-local Phase 2 settings; infrastructure remains replaceable."""

    max_upload_bytes: int
    request_store_max_entries: int
    cache_max_entries: int
    log_buffer_max_entries: int
    pipeline_version: str
    schema_version: str
    default_owner_id: UUID

    @classmethod
    def from_environment(cls) -> "Settings":
        pipeline_version = os.getenv("OCR_PIPELINE_VERSION", "phase2-ocr-v1").strip()
        schema_version = os.getenv("OCR_SCHEMA_VERSION", "ocr-response-v1").strip()
        if not pipeline_version or not schema_version:
            raise ValueError("OCR pipeline and schema versions must not be blank")
        return cls(
            max_upload_bytes=_positive_int("OCR_MAX_UPLOAD_BYTES", 25 * 1024 * 1024),
            request_store_max_entries=_positive_int("OCR_REQUEST_STORE_MAX_ENTRIES", 512),
            cache_max_entries=_positive_int("OCR_CACHE_MAX_ENTRIES", 128),
            log_buffer_max_entries=_positive_int("OCR_LOG_BUFFER_MAX_ENTRIES", 500),
            pipeline_version=pipeline_version,
            schema_version=schema_version,
            default_owner_id=UUID(
                os.getenv("OCR_DEFAULT_OWNER_ID", "00000000-0000-0000-0000-000000000001")
            ),
        )


settings = Settings.from_environment()

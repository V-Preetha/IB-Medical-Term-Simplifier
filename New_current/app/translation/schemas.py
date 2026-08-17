"""Versioned translation schemas."""

from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.translation.provider import SUPPORTED_LANGUAGES


class TranslationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200_000)
    source_language: str = Field(default="eng_Latn", pattern="^eng_Latn$")
    target_language: str = Field(examples=list(SUPPORTED_LANGUAGES))

    @field_validator("target_language")
    @classmethod
    def supported_target(cls, value: str) -> str:
        if value not in SUPPORTED_LANGUAGES:
            raise ValueError("target_language is not enabled for the MVP")
        return value


class TranslationResponse(BaseModel):
    schema_version: str = "translation-response-v1"
    pipeline_version: str = "mvp-v1"
    request_id: UUID
    source_language: str
    target_language: str
    translated_text: str
    provider_name: str
    model_name: str
    model_version: str
    confidence: float | None = None
    confidence_method: str = "not_available_model_has_no_native_calibrated_quality_score"
    calibration_version: str = "not_calibrated-v1"
    processing_time_ms: float
    cache_hit: bool = False
    review_required: bool = True
    preservation_policy: str = "numeric-unit-bracketed-placeholders-v2"
    warnings: list[str] = Field(
        default_factory=lambda: [
            "Medical verification and translation quality calibration are deferred for MVP."
        ]
    )


class TranslationModelResponse(BaseModel):
    status: str
    provider_name: str
    model_name: str
    model_revision: str
    device: str
    detail: str
    supported_languages: dict[str, str]
    configuration: dict[str, object]

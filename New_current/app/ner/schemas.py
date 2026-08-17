"""Versioned production medical NER schemas."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ner.contracts import ENTITY_TYPES


class EntitySchema(BaseModel):
    text: str = Field(min_length=1)
    label: str = Field(examples=list(ENTITY_TYPES))
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_span(self) -> "EntitySchema":
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        if self.label not in ENTITY_TYPES:
            raise ValueError("label must be a canonical medical entity type")
        return self


class NERRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"text": "The patient takes metformin for type 2 diabetes."}}
    )

    text: str = Field(
        min_length=1,
        max_length=200_000,
        description="Normalized OCR text. Clinical text is not written to normal logs.",
    )


class NERInferenceMetadataSchema(BaseModel):
    framework: str
    device: str
    token_count: int | None = Field(default=None, ge=0)
    tokens_per_second: float | None = Field(default=None, ge=0)
    model_loading_time_ms: float | None = Field(default=None, ge=0)
    startup_timestamp: datetime | None
    configuration: dict[str, Any]


class NERResponse(BaseModel):
    schema_version: str = "ner-response-v1"
    pipeline_version: str = "phase5-ner-v1"
    request_id: UUID
    text: str
    entities: list[EntitySchema]
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_method: str = "mean_entity_softmax_probability"
    calibration_version: str = "uncalibrated-biomedical-ner-all-v1"
    review_required: bool
    processing_time_ms: float = Field(ge=0)
    cache_hit: bool = False
    provider_name: str
    model_name: str
    model_revision: str
    inference_metadata: NERInferenceMetadataSchema
    warnings: list[str]


class NERModelSchema(BaseModel):
    provider_name: str
    model_name: str
    model_revision: str
    framework: str
    status: str
    detail: str
    device: str
    loading_time_ms: float | None
    startup_timestamp: datetime | None
    configuration: dict[str, Any]


class NERModelsResponse(BaseModel):
    models: list[NERModelSchema]


class NERHealthResponse(BaseModel):
    status: str
    providers: list[NERModelSchema]


class NERErrorDetail(BaseModel):
    code: str
    message: str


class NERErrorResponse(BaseModel):
    error: NERErrorDetail
    request_id: str | None = None

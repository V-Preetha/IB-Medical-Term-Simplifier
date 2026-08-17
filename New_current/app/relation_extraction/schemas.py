"""Versioned relation-extraction HTTP schemas."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RelationEntitySchema(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)
    label: str = Field(min_length=1, max_length=100)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    confidence: float = Field(ge=0, le=1)
    concept_id: str | None = Field(default=None, max_length=100)
    preferred_name: str | None = Field(default=None, max_length=1_000)

    @model_validator(mode="after")
    def validate_offsets(self) -> "RelationEntitySchema":
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class RelationExtractionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Metformin treats diabetes. HbA1c indicates diabetes.",
                "entities": [
                    {
                        "text": "Metformin",
                        "label": "Medication",
                        "start": 0,
                        "end": 9,
                        "confidence": 0.98,
                    },
                    {
                        "text": "diabetes",
                        "label": "Disease",
                        "start": 17,
                        "end": 25,
                        "confidence": 0.97,
                    },
                ],
            }
        }
    )
    text: str = Field(min_length=1, max_length=200_000)
    entities: list[RelationEntitySchema] = Field(min_length=2, max_length=100)


class RelationInferenceMetadataSchema(BaseModel):
    confidence_method: str
    calibration_version: str
    preprocessing_version: str


class ClinicalRelationSchema(BaseModel):
    source: RelationEntitySchema
    target: RelationEntitySchema
    relation: str
    confidence: float = Field(ge=0, le=1)
    evidence_start: int = Field(ge=0)
    evidence_end: int = Field(gt=0)
    model_name: str
    model_revision: str
    inference_metadata: RelationInferenceMetadataSchema


class RelationReproducibilitySchema(BaseModel):
    provider_name: str
    provider_version: str
    model_name: str
    model_revision: str
    framework: str
    device: str
    relation_labels: list[str]
    confidence_method: str
    calibration_version: str
    preprocessing_version: str
    startup_timestamp: datetime | None
    loading_time_ms: float | None
    configuration: dict[str, Any]


class RelationExtractionResponse(BaseModel):
    schema_version: str = "relation-extraction-response-v1"
    pipeline_version: str = "phase7-relation-extraction-v1"
    request_id: UUID
    relations: list[ClinicalRelationSchema]
    processing_time_ms: float = Field(ge=0)
    candidate_pair_count: int = Field(ge=0)
    token_count: int = Field(ge=0)
    tokens_per_second: float | None = Field(default=None, ge=0)
    cache_hit: bool = False
    reproducibility: RelationReproducibilitySchema
    warnings: list[str]


class RelationModelSchema(BaseModel):
    provider_name: str
    provider_version: str
    model_name: str
    model_revision: str
    framework: str
    device: str
    relation_labels: list[str]
    status: str
    detail: str
    loading_time_ms: float | None
    startup_timestamp: datetime | None
    configuration: dict[str, Any]


class RelationModelsResponse(BaseModel):
    models: list[RelationModelSchema]


class RelationHealthResponse(BaseModel):
    status: str
    providers: list[RelationModelSchema]


class RelationErrorDetail(BaseModel):
    code: str
    message: str


class RelationErrorResponse(BaseModel):
    error: RelationErrorDetail
    request_id: str | None = None

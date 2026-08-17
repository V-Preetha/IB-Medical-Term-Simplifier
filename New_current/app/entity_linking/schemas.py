"""Versioned HTTP schemas for entity linking."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SourceEntitySchema(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)
    label: str = Field(min_length=1, max_length=100)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_span(self) -> "SourceEntitySchema":
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class EntityLinkingRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "entities": [
                    {
                        "text": "diabetes mellitus",
                        "label": "Disease",
                        "start": 24,
                        "end": 41,
                        "confidence": 0.98,
                    }
                ]
            }
        }
    )
    entities: list[SourceEntitySchema] = Field(min_length=1, max_length=1_000)


class ConceptCandidateSchema(BaseModel):
    concept_id: str
    preferred_name: str
    semantic_types: list[str]
    confidence: float = Field(ge=0, le=1)
    source_ontology: str


class EntityLinkSchema(BaseModel):
    original_entity: SourceEntitySchema
    status: str
    normalized_concept: ConceptCandidateSchema | None
    candidates: list[ConceptCandidateSchema]
    requires_review: bool


class ReproducibilityMetadataSchema(BaseModel):
    provider_name: str
    provider_version: str
    model_name: str
    model_version: str
    terminology_name: str
    terminology_version: str
    confidence_method: str
    calibration_version: str
    loading_time_ms: float | None
    startup_timestamp: datetime | None
    configuration: dict[str, Any]


class EntityLinkingResponse(BaseModel):
    schema_version: str = "entity-linking-response-v1"
    pipeline_version: str = "phase6-entity-linking-v1"
    request_id: UUID
    links: list[EntityLinkSchema]
    processing_time_ms: float = Field(ge=0)
    cache_hit: bool = False
    reproducibility: ReproducibilityMetadataSchema
    warnings: list[str]


class EntityLinkingModelSchema(BaseModel):
    provider_name: str
    provider_version: str
    model_name: str
    model_version: str
    terminology_name: str
    terminology_version: str
    status: str
    detail: str
    loading_time_ms: float | None
    startup_timestamp: datetime | None
    configuration: dict[str, Any]


class EntityLinkingModelsResponse(BaseModel):
    models: list[EntityLinkingModelSchema]


class EntityLinkingHealthResponse(BaseModel):
    status: str
    providers: list[EntityLinkingModelSchema]


class EntityLinkingErrorDetail(BaseModel):
    code: str
    message: str


class EntityLinkingErrorResponse(BaseModel):
    error: EntityLinkingErrorDetail
    request_id: str | None = None

"""Versioned medical embedding HTTP schemas."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EmbeddingInputSchema(BaseModel):
    input_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=200_000)


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "inputs": [
                    {
                        "input_id": "segment-1",
                        "text": "Metformin is prescribed for type 2 diabetes.",
                    },
                    {
                        "input_id": "segment-2",
                        "text": "HbA1c was measured during follow-up.",
                    },
                ]
            }
        }
    )
    inputs: list[EmbeddingInputSchema] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_unique_identifiers(self) -> "EmbeddingRequest":
        identifiers = [item.input_id for item in self.inputs]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("input_id values must be unique")
        return self


class EmbeddingVectorSchema(BaseModel):
    input_id: str
    vector: list[float]
    dimensions: int = Field(gt=0)
    token_count: int = Field(ge=0)
    vector_norm: float = Field(ge=0)


class EmbeddingReproducibilitySchema(BaseModel):
    provider_name: str
    provider_version: str
    model_name: str
    model_revision: str
    framework: str
    device: str
    dimensions: int | None
    pooling_method: str
    normalized: bool
    startup_timestamp: datetime | None
    loading_time_ms: float | None
    configuration: dict[str, Any]


class EmbeddingResponse(BaseModel):
    schema_version: str = "medical-embedding-response-v1"
    pipeline_version: str = "phase8-medical-embeddings-v1"
    request_id: UUID
    embeddings: list[EmbeddingVectorSchema]
    batch_size: int = Field(gt=0)
    processing_time_ms: float = Field(ge=0)
    tokens_per_second: float | None = Field(default=None, ge=0)
    cache_hit: bool = False
    confidence: None = None
    confidence_method: str = "not_applicable_vector_representation"
    reproducibility: EmbeddingReproducibilitySchema
    warnings: list[str]


class EmbeddingModelSchema(BaseModel):
    provider_name: str
    provider_version: str
    model_name: str
    model_revision: str
    framework: str
    device: str
    dimensions: int | None
    pooling_method: str
    normalized: bool
    status: str
    detail: str
    loading_time_ms: float | None
    startup_timestamp: datetime | None
    configuration: dict[str, Any]


class EmbeddingModelsResponse(BaseModel):
    models: list[EmbeddingModelSchema]


class EmbeddingHealthResponse(BaseModel):
    status: str
    providers: list[EmbeddingModelSchema]


class EmbeddingErrorDetail(BaseModel):
    code: str
    message: str


class EmbeddingErrorResponse(BaseModel):
    error: EmbeddingErrorDetail
    request_id: str | None = None

"""Versioned Medical Verification API schemas."""

from uuid import UUID

from pydantic import BaseModel, Field


class VerificationRequest(BaseModel):
    premise: str = Field(min_length=1, max_length=200_000, examples=["No pneumonia."])
    hypothesis: str = Field(min_length=1, max_length=200_000, examples=["There is no pneumonia."])


class VerificationResponse(BaseModel):
    schema_version: str = "medical-verification-v1"
    request_id: UUID
    label: str
    probabilities: dict[str, float]
    verification: str
    review_required: bool
    deterministic_mismatches: list[dict[str, object]]
    processing_time_ms: float
    nli_latency_ms: float
    model_name: str
    model_revision: str


class VerificationModelResponse(BaseModel):
    status: str
    model_name: str
    model_revision: str
    device: str
    detail: str
    configuration: dict[str, object]

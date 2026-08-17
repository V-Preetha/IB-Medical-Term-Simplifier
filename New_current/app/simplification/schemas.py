"""Versioned Phase 9 medical simplification API schemas."""

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.ner.schemas import EntitySchema


class LinkedConceptSchema(BaseModel):
    entity_text: str = Field(min_length=1, max_length=500)
    concept_id: str = Field(min_length=1, max_length=100)
    preferred_name: str = Field(min_length=1, max_length=500)
    semantic_type: str | None = Field(default=None, max_length=200)


class SimplificationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200_000)
    entities: list[EntitySchema] = Field(default_factory=list, max_length=5_000)
    linked_concepts: list[LinkedConceptSchema] = Field(default_factory=list, max_length=5_000)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "HbA1c was 7.2%. Metformin was continued for type 2 diabetes.",
                    "entities": [
                        {
                            "text": "Metformin",
                            "label": "Medication",
                            "start": 15,
                            "end": 24,
                            "confidence": 0.98,
                        }
                    ],
                    "linked_concepts": [],
                }
            ]
        }
    }


class MedicalTermExplanationSchema(BaseModel):
    term: str
    explanation: str


class SimplifiedLevelSchema(BaseModel):
    level: Literal["clinical", "general_public", "child_friendly"]
    original_report: str
    simplified_report: str
    medical_terms_explained: list[MedicalTermExplanationSchema]
    important_findings: list[str]
    suggested_questions_for_doctor: list[str]
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_method: str = "source-fact-preservation-ratio-v1"
    calibration_version: str = "uncalibrated-fidelity-proxy-v1"
    processing_time_ms: float = Field(ge=0)
    model_revision: str
    pipeline_version: str
    prompt_version: str
    review_required: bool
    warnings: list[str]


class InferenceMetadataSchema(BaseModel):
    provider_name: str
    model_name: str
    model_revision: str
    device: str
    prompt_version: str
    prompt_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    generation_time_ms: float = Field(ge=0)
    deterministic_generation: bool
    local_files_only: bool


class SimplificationResponse(BaseModel):
    schema_version: str = "simplification-response-v2"
    pipeline_version: str = "medical-simplification-v1"
    request_id: UUID
    clinical: SimplifiedLevelSchema
    general_public: SimplifiedLevelSchema
    child_friendly: SimplifiedLevelSchema
    processing_time_ms: float = Field(ge=0)
    cache_hit: bool = False
    inference: InferenceMetadataSchema


class LegacySimplificationSectionsSchema(BaseModel):
    executive_summary: str
    important_findings: list[str]
    timeline: list[str]
    medical_terms_explained: list[str]
    simple_explanation: str
    recommended_follow_up: list[str]


class LegacySimplificationResponse(BaseModel):
    schema_version: str = "simplification-response-v1"
    pipeline_version: str = "mvp-v1"
    request_id: UUID
    simplified_report: str
    sections: LegacySimplificationSectionsSchema
    model_name: str
    model_version: str
    provider_name: str
    prompt_version: str
    confidence: float | None = None
    confidence_method: str = "source-fact-preservation-ratio-v1"
    calibration_version: str = "uncalibrated-fidelity-proxy-v1"
    processing_time_ms: float
    cache_hit: bool = False
    review_required: bool = True
    warnings: list[str]


class StageModelResponse(BaseModel):
    status: Literal["HEALTHY", "UNAVAILABLE"]
    provider_name: str
    model_name: str
    model_revision: str
    prompt_version: str
    device: str
    detail: str
    model_loading_time_ms: float | None
    configuration: dict[str, Any]

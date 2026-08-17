"""Production BioLinkBERT relation-extraction endpoints."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, Response, status

from app.relation_extraction.contracts import (
    ClinicalRelation,
    RelationDocument,
    RelationEntity,
    RelationHealthStatus,
    RelationProviderHealth,
)
from app.relation_extraction.schemas import (
    ClinicalRelationSchema,
    RelationEntitySchema,
    RelationErrorResponse,
    RelationExtractionRequest,
    RelationExtractionResponse,
    RelationHealthResponse,
    RelationInferenceMetadataSchema,
    RelationModelSchema,
    RelationModelsResponse,
    RelationReproducibilitySchema,
)
from app.relation_extraction.service import RelationExtractionService

router = APIRouter(prefix="/api/v1/relation-extraction", tags=["Relation Extraction"])


def get_relation_extraction_service(request: Request) -> RelationExtractionService:
    return request.app.state.relation_extraction_service


ServiceDependency = Annotated[RelationExtractionService, Depends(get_relation_extraction_service)]


@router.post(
    "",
    response_model=RelationExtractionResponse,
    summary="Extract biomedical relations",
    description=(
        "Classifies directed relationships between supplied medical NER entities using "
        "the approved immutable local BioLinkBERT relation checkpoint. Optional UMLS "
        "identifiers are preserved when available."
    ),
    responses={
        200: {
            "description": "Relations extracted with measured probabilities and provenance.",
            "content": {
                "application/json": {
                    "example": {
                        "schema_version": "relation-extraction-response-v1",
                        "pipeline_version": "phase7-relation-extraction-v1",
                        "request_id": "fbeac72f-4cf4-46fd-8a53-211441661cbc",
                        "relations": [
                            {
                                "source": {
                                    "text": "Metformin",
                                    "label": "Medication",
                                    "start": 0,
                                    "end": 9,
                                    "confidence": 0.98,
                                },
                                "target": {
                                    "text": "diabetes",
                                    "label": "Disease",
                                    "start": 17,
                                    "end": 25,
                                    "confidence": 0.97,
                                },
                                "relation": "TREATS",
                                "confidence": 0.94,
                                "evidence_start": 0,
                                "evidence_end": 25,
                                "model_name": "michiyasunaga/BioLinkBERT-base",
                                "model_revision": "immutable-commit-sha",
                                "inference_metadata": {
                                    "confidence_method": "softmax_relation_class_probability",
                                    "calibration_version": "uncalibrated-biolinkbert-re-v1",
                                    "preprocessing_version": "entity-marker-v1",
                                },
                            }
                        ],
                        "processing_time_ms": 21.4,
                        "candidate_pair_count": 2,
                        "token_count": 38,
                        "tokens_per_second": 1775.7,
                        "cache_hit": False,
                        "reproducibility": {},
                        "warnings": [],
                    }
                }
            },
        },
        422: {"model": RelationErrorResponse, "description": "Invalid input schema."},
        500: {"model": RelationErrorResponse, "description": "Inference failed."},
        503: {
            "model": RelationErrorResponse,
            "description": "Approved relation-classification artifact unavailable.",
        },
    },
)
async def extract_relations(
    payload: RelationExtractionRequest,
    request: Request,
    service: ServiceDependency,
) -> RelationExtractionResponse:
    result = await service.process(
        RelationDocument(
            text=payload.text,
            entities=tuple(_domain_entity(entity) for entity in payload.entities),
        ),
        request.state.request_id,
    )
    metadata = result.metadata
    inference = RelationInferenceMetadataSchema(
        confidence_method=str(metadata["confidence_method"]),
        calibration_version=str(metadata["calibration_version"]),
        preprocessing_version=str(metadata["preprocessing_version"]),
    )
    return RelationExtractionResponse(
        request_id=result.request_id,
        relations=[
            _relation_schema(relation, metadata, inference) for relation in result.relations
        ],
        processing_time_ms=result.processing_time_ms,
        candidate_pair_count=result.candidate_pair_count,
        token_count=result.token_count,
        tokens_per_second=result.tokens_per_second,
        reproducibility=RelationReproducibilitySchema.model_validate(metadata),
        warnings=list(result.warnings),
    )


@router.get(
    "/models",
    response_model=RelationModelsResponse,
    summary="Get the production relation model",
    description=(
        "Returns the approved BioLinkBERT checkpoint, dynamic relation ontology, device, "
        "configuration, and readiness state."
    ),
    responses={200: {"description": "Production model inventory returned."}},
)
async def production_models(service: ServiceDependency) -> RelationModelsResponse:
    return RelationModelsResponse(models=[_model_schema(service.health())])


@router.get(
    "/health",
    response_model=RelationHealthResponse,
    summary="Check relation-extraction readiness",
    description=(
        "Validates immutable local BioLinkBERT artifacts and relation-head compatibility "
        "without running clinical inference."
    ),
    responses={
        200: {"description": "BioLinkBERT relation provider is ready."},
        503: {"description": "Checkpoint is unavailable, incompatible, or not initialized."},
    },
)
async def production_health(
    response: Response, service: ServiceDependency
) -> RelationHealthResponse:
    health = service.health()
    healthy = health.status is RelationHealthStatus.READY
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return RelationHealthResponse(
        status="HEALTHY" if healthy else "NOT_READY",
        providers=[_model_schema(health)],
    )


def _domain_entity(value: RelationEntitySchema) -> RelationEntity:
    return RelationEntity(
        text=value.text,
        label=value.label,
        start=value.start,
        end=value.end,
        confidence=value.confidence,
        concept_id=value.concept_id,
        preferred_name=value.preferred_name,
    )


def _relation_schema(
    relation: ClinicalRelation,
    metadata: dict[str, object],
    inference: RelationInferenceMetadataSchema,
) -> ClinicalRelationSchema:
    return ClinicalRelationSchema(
        source=RelationEntitySchema.model_validate(relation.source, from_attributes=True),
        target=RelationEntitySchema.model_validate(relation.target, from_attributes=True),
        relation=relation.relation_type,
        confidence=relation.confidence,
        evidence_start=relation.evidence_start,
        evidence_end=relation.evidence_end,
        model_name=str(metadata["model_name"]),
        model_revision=str(metadata["model_revision"]),
        inference_metadata=inference,
    )


def _model_schema(health: RelationProviderHealth) -> RelationModelSchema:
    metadata = health.metadata
    return RelationModelSchema(
        provider_name=metadata.provider_name,
        provider_version=metadata.provider_version,
        model_name=metadata.model_name,
        model_revision=metadata.model_revision,
        framework=metadata.framework,
        device=metadata.device,
        relation_labels=list(metadata.relation_labels),
        status=health.status.value,
        detail=health.detail,
        loading_time_ms=_optional_float(metadata.loading_time_ms),
        startup_timestamp=metadata.startup_timestamp,
        configuration=metadata.configuration,
    )


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None

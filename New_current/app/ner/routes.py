"""Production HTTP endpoints for the approved medical NER model."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, Response, status

from app.ner.contracts import NERHealthStatus, NERProviderHealth
from app.ner.schemas import (
    EntitySchema,
    NERErrorResponse,
    NERHealthResponse,
    NERInferenceMetadataSchema,
    NERModelSchema,
    NERModelsResponse,
    NERRequest,
    NERResponse,
)
from app.ner.service import MedicalNERService

router = APIRouter(prefix="/api/v1/ner", tags=["Medical NER"])


def get_ner_service(request: Request) -> MedicalNERService:
    return request.app.state.ner_service


NERServiceDependency = Annotated[MedicalNERService, Depends(get_ner_service)]


@router.post(
    "",
    response_model=NERResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract medical entities",
    description=(
        "Runs the approved immutable biomedical-ner-all provider on normalized OCR text "
        "and returns canonical entity labels, exact character offsets, measured token "
        "probabilities, latency, and reproducibility metadata."
    ),
    responses={
        200: {
            "description": "Medical entities extracted successfully.",
            "content": {
                "application/json": {
                    "example": {
                        "schema_version": "ner-response-v1",
                        "pipeline_version": "phase5-ner-v1",
                        "request_id": "fbeac72f-4cf4-46fd-8a53-211441661cbc",
                        "text": "The patient takes metformin for diabetes.",
                        "entities": [
                            {
                                "text": "metformin",
                                "label": "Medication",
                                "start": 18,
                                "end": 27,
                                "confidence": 0.99,
                            }
                        ],
                        "confidence": 0.99,
                        "confidence_method": "mean_entity_softmax_probability",
                        "calibration_version": "uncalibrated-biomedical-ner-all-v1",
                        "review_required": False,
                        "processing_time_ms": 24.5,
                        "cache_hit": False,
                        "provider_name": "biomedical-ner-all",
                        "model_name": "biomedical-ner-all",
                        "model_revision": "015a4050c9ac99722e61c547aa9b4282bcbedc7f",
                        "inference_metadata": {
                            "framework": "transformers-token-classification",
                            "device": "cpu",
                            "token_count": 11,
                            "tokens_per_second": 448.98,
                            "model_loading_time_ms": 125.386,
                            "startup_timestamp": "2026-08-04T18:00:00Z",
                            "configuration": {},
                        },
                        "warnings": [],
                    }
                }
            },
        },
        422: {"model": NERErrorResponse, "description": "Invalid input schema."},
        500: {"model": NERErrorResponse, "description": "NER inference failed."},
        503: {"model": NERErrorResponse, "description": "NER provider unavailable."},
    },
)
async def extract_entities(
    payload: NERRequest,
    request: Request,
    service: NERServiceDependency,
) -> NERResponse:
    result = await service.process(payload.text, request.state.request_id)
    threshold = float(result.metadata["configuration"]["confidence_threshold"])
    return NERResponse(
        request_id=result.request_id,
        text=payload.text,
        entities=[
            EntitySchema.model_validate(entity, from_attributes=True) for entity in result.entities
        ],
        confidence=result.confidence,
        review_required=result.confidence is None or result.confidence < threshold,
        processing_time_ms=result.processing_time_ms,
        provider_name=result.provider_name,
        model_name=result.model_name,
        model_revision=result.model_revision,
        inference_metadata=NERInferenceMetadataSchema(
            framework=str(result.metadata["framework"]),
            device=result.device,
            token_count=result.token_count,
            tokens_per_second=result.tokens_per_second,
            model_loading_time_ms=_optional_float(result.metadata["loading_time_ms"]),
            startup_timestamp=result.metadata["startup_timestamp"],
            configuration=result.metadata["configuration"],
        ),
        warnings=list(result.warnings),
    )


@router.get(
    "/models",
    response_model=NERModelsResponse,
    summary="Get the production NER model",
    description=(
        "Returns metadata for the single approved production NER provider. Archived "
        "benchmark candidates are deliberately excluded."
    ),
    responses={200: {"description": "Approved production model returned."}},
)
async def production_model(service: NERServiceDependency) -> NERModelsResponse:
    return NERModelsResponse(models=[_model_schema(service.health())])


@router.get(
    "/health",
    response_model=NERHealthResponse,
    summary="Check production NER readiness",
    description=(
        "Reports readiness for the approved biomedical-ner-all checkpoint without running "
        "clinical inference."
    ),
    responses={
        200: {"description": "Production NER provider is healthy."},
        503: {"description": "Production NER provider is unavailable."},
    },
)
async def production_health(response: Response, service: NERServiceDependency) -> NERHealthResponse:
    provider = service.health()
    healthy = provider.status is NERHealthStatus.READY
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return NERHealthResponse(
        status="HEALTHY" if healthy else "UNHEALTHY",
        providers=[_model_schema(provider)],
    )


def _model_schema(health: NERProviderHealth) -> NERModelSchema:
    metadata = health.metadata
    return NERModelSchema(
        provider_name=health.provider_name,
        model_name=metadata.model_name,
        model_revision=metadata.model_revision,
        framework=metadata.framework,
        status=health.status.value,
        detail=health.detail,
        device=metadata.device,
        loading_time_ms=metadata.loading_time_ms,
        startup_timestamp=metadata.startup_timestamp,
        configuration=metadata.configuration,
    )


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None

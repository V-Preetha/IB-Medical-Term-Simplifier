"""Production medical embedding endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status

from app.embeddings.contracts import (
    EmbeddingHealthStatus,
    EmbeddingInput,
    EmbeddingProviderHealth,
)
from app.embeddings.schemas import (
    EmbeddingErrorResponse,
    EmbeddingHealthResponse,
    EmbeddingModelSchema,
    EmbeddingModelsResponse,
    EmbeddingReproducibilitySchema,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVectorSchema,
)
from app.embeddings.service import MedicalEmbeddingService

router = APIRouter(prefix="/api/v1/embeddings", tags=["Medical Embeddings"])


def get_embedding_service(request: Request) -> MedicalEmbeddingService:
    return request.app.state.embedding_service


ServiceDependency = Annotated[MedicalEmbeddingService, Depends(get_embedding_service)]


@router.post(
    "",
    response_model=EmbeddingResponse,
    summary="Generate medical text embeddings",
    description=(
        "Generates a batch of normalized BioClinical ModernBERT vectors from supplied "
        "medical text. It returns vectors directly and does not write to a vector store."
    ),
    responses={
        200: {"description": "Embedding batch generated successfully."},
        422: {"model": EmbeddingErrorResponse, "description": "Invalid batch input."},
        500: {"model": EmbeddingErrorResponse, "description": "Embedding failed."},
        503: {
            "model": EmbeddingErrorResponse,
            "description": "Approved local embedding model unavailable.",
        },
    },
)
async def generate_embeddings(
    payload: EmbeddingRequest,
    request: Request,
    service: ServiceDependency,
) -> EmbeddingResponse:
    result = await service.process(
        tuple(EmbeddingInput(item.input_id, item.text) for item in payload.inputs),
        request.state.request_id,
    )
    return EmbeddingResponse(
        request_id=result.request_id,
        embeddings=[
            EmbeddingVectorSchema(
                input_id=item.input_id,
                vector=list(item.values),
                dimensions=len(item.values),
                token_count=item.token_count,
                vector_norm=item.vector_norm,
            )
            for item in result.embeddings
        ],
        batch_size=len(result.embeddings),
        processing_time_ms=result.processing_time_ms,
        tokens_per_second=result.tokens_per_second,
        reproducibility=EmbeddingReproducibilitySchema.model_validate(result.metadata),
        warnings=list(result.warnings),
    )


@router.get(
    "/models",
    response_model=EmbeddingModelsResponse,
    summary="Get the medical embedding model",
    description=(
        "Returns the exact BioClinical ModernBERT identity, pooling contract, vector "
        "dimensions, device, configuration, and readiness."
    ),
    responses={200: {"description": "Embedding model inventory returned."}},
)
async def production_models(service: ServiceDependency) -> EmbeddingModelsResponse:
    return EmbeddingModelsResponse(models=[_model_schema(service.health())])


@router.get(
    "/health",
    response_model=EmbeddingHealthResponse,
    summary="Check medical embedding readiness",
    description=(
        "Checks immutable local BioClinical ModernBERT availability without generating "
        "an embedding."
    ),
    responses={
        200: {"description": "Embedding provider is ready."},
        503: {"description": "Approved local model is unavailable or not configured."},
    },
)
async def production_health(
    response: Response, service: ServiceDependency
) -> EmbeddingHealthResponse:
    health = service.health()
    healthy = health.status is EmbeddingHealthStatus.READY
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return EmbeddingHealthResponse(
        status="HEALTHY" if healthy else "NOT_READY",
        providers=[_model_schema(health)],
    )


def _model_schema(health: EmbeddingProviderHealth) -> EmbeddingModelSchema:
    metadata = health.metadata
    return EmbeddingModelSchema(
        provider_name=metadata.provider_name,
        provider_version=metadata.provider_version,
        model_name=metadata.model_name,
        model_revision=metadata.model_revision,
        framework=metadata.framework,
        device=metadata.device,
        dimensions=metadata.dimensions,
        pooling_method=metadata.pooling_method,
        normalized=metadata.normalized,
        status=health.status.value,
        detail=health.detail,
        loading_time_ms=metadata.loading_time_ms,
        startup_timestamp=metadata.startup_timestamp,
        configuration=metadata.configuration,
    )

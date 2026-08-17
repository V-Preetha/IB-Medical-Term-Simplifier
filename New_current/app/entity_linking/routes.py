"""Production entity-linking endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status

from app.entity_linking.contracts import (
    ConceptCandidate,
    EntityLink,
    LinkerHealth,
    LinkerHealthStatus,
    SourceEntity,
)
from app.entity_linking.schemas import (
    ConceptCandidateSchema,
    EntityLinkingErrorResponse,
    EntityLinkingHealthResponse,
    EntityLinkingModelSchema,
    EntityLinkingModelsResponse,
    EntityLinkingRequest,
    EntityLinkingResponse,
    EntityLinkSchema,
    ReproducibilityMetadataSchema,
    SourceEntitySchema,
)
from app.entity_linking.service import EntityLinkingService

router = APIRouter(prefix="/api/v1/entity-linking", tags=["Entity Linking"])


def get_entity_linking_service(request: Request) -> EntityLinkingService:
    return request.app.state.entity_linking_service


ServiceDependency = Annotated[EntityLinkingService, Depends(get_entity_linking_service)]


@router.post(
    "",
    response_model=EntityLinkingResponse,
    summary="Link medical entities to UMLS concepts",
    description=(
        "Links normalized NER spans to canonical UMLS concepts using the approved local "
        "SciSpaCy provider. Clinical text is not written to normal logs."
    ),
    responses={
        200: {"description": "Entities linked; ambiguous and unlinked spans remain explicit."},
        422: {"model": EntityLinkingErrorResponse, "description": "Invalid input."},
        500: {"model": EntityLinkingErrorResponse, "description": "Linking failed."},
        503: {
            "model": EntityLinkingErrorResponse,
            "description": "Licensed local UMLS provider unavailable.",
        },
    },
)
async def link_entities(
    payload: EntityLinkingRequest,
    request: Request,
    service: ServiceDependency,
) -> EntityLinkingResponse:
    result = await service.process(
        tuple(
            SourceEntity(
                text=item.text,
                label=item.label,
                start=item.start,
                end=item.end,
                confidence=item.confidence,
            )
            for item in payload.entities
        ),
        request.state.request_id,
    )
    return EntityLinkingResponse(
        request_id=result.request_id,
        links=[_link_schema(link) for link in result.links],
        processing_time_ms=result.processing_time_ms,
        reproducibility=ReproducibilityMetadataSchema.model_validate(result.metadata),
        warnings=list(result.warnings),
    )


@router.get(
    "/models",
    response_model=EntityLinkingModelsResponse,
    summary="Get the production entity-linking model",
    description="Returns the configured SciSpaCy language model and UMLS release identity.",
    responses={200: {"description": "Provider inventory returned."}},
)
async def production_models(
    service: ServiceDependency,
) -> EntityLinkingModelsResponse:
    return EntityLinkingModelsResponse(models=[_model_schema(service.health())])


@router.get(
    "/health",
    response_model=EntityLinkingHealthResponse,
    summary="Check entity-linking readiness",
    description=(
        "Checks the approved local SciSpaCy model and licensed UMLS resources without "
        "performing inference."
    ),
    responses={
        200: {"description": "Entity-linking provider is ready."},
        503: {"description": "Provider configuration or local resources are unavailable."},
    },
)
async def production_health(
    response: Response, service: ServiceDependency
) -> EntityLinkingHealthResponse:
    health = service.health()
    healthy = health.status is LinkerHealthStatus.READY
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return EntityLinkingHealthResponse(
        status="HEALTHY" if healthy else "NOT_READY",
        providers=[_model_schema(health)],
    )


def _candidate_schema(candidate: ConceptCandidate) -> ConceptCandidateSchema:
    return ConceptCandidateSchema(
        concept_id=candidate.concept_id,
        preferred_name=candidate.preferred_name,
        semantic_types=list(candidate.semantic_types),
        confidence=candidate.confidence,
        source_ontology=candidate.source_ontology,
    )


def _link_schema(link: EntityLink) -> EntityLinkSchema:
    source = SourceEntitySchema.model_validate(link.original_entity, from_attributes=True)
    return EntityLinkSchema(
        original_entity=source,
        status=link.status.value,
        normalized_concept=(
            _candidate_schema(link.selected_concept) if link.selected_concept is not None else None
        ),
        candidates=[_candidate_schema(candidate) for candidate in link.candidates],
        requires_review=link.requires_review,
    )


def _model_schema(health: LinkerHealth) -> EntityLinkingModelSchema:
    metadata = health.metadata
    return EntityLinkingModelSchema(
        provider_name=metadata.provider_name,
        provider_version=metadata.provider_version,
        model_name=metadata.model_name,
        model_version=metadata.model_version,
        terminology_name=metadata.terminology_name,
        terminology_version=metadata.terminology_version,
        status=health.status.value,
        detail=health.detail,
        loading_time_ms=metadata.loading_time_ms,
        startup_timestamp=metadata.startup_timestamp,
        configuration=metadata.configuration,
    )

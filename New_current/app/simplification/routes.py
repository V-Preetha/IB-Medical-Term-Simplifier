"""Phase 9 Qwen3 medical report simplification HTTP API."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status

from app.simplification.schemas import (
    InferenceMetadataSchema,
    LegacySimplificationResponse,
    LegacySimplificationSectionsSchema,
    MedicalTermExplanationSchema,
    SimplificationRequest,
    SimplificationResponse,
    SimplifiedLevelSchema,
    StageModelResponse,
)
from app.simplification.service import SimplificationResult, SimplificationService, ValidatedLevel

router = APIRouter(tags=["Medical Report Simplification"])


def get_service(request: Request) -> SimplificationService:
    return request.app.state.simplification_service


Dependency = Annotated[SimplificationService, Depends(get_service)]


def _group_entities(payload: SimplificationRequest) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, tuple[str, ...]] = {}
    for entity in payload.entities:
        grouped[entity.label] = (*grouped.get(entity.label, ()), entity.text)
    return grouped


async def _process(
    payload: SimplificationRequest, request: Request, service: SimplificationService
) -> SimplificationResult:
    linked = tuple(item.model_dump(exclude_none=True) for item in payload.linked_concepts)
    return await service.process(
        payload.text,
        _group_entities(payload),
        request.state.request_id,
        linked,
    )


def _level_schema(
    value: ValidatedLevel,
    result: SimplificationResult,
    service: SimplificationService,
) -> SimplifiedLevelSchema:
    metadata = service.provider.metadata()
    level = value.result
    return SimplifiedLevelSchema(
        level=level.level,
        original_report=result.original_report,
        simplified_report=level.simplified_report,
        medical_terms_explained=[
            MedicalTermExplanationSchema(term=item.term, explanation=item.explanation)
            for item in level.medical_terms_explained
        ],
        important_findings=list(level.important_findings),
        suggested_questions_for_doctor=list(level.suggested_questions_for_doctor),
        confidence=value.confidence,
        processing_time_ms=result.processing_time_ms,
        model_revision=metadata.model_revision,
        pipeline_version="medical-simplification-v1",
        prompt_version=metadata.prompt_version,
        review_required=value.review_required,
        warnings=list(value.warnings),
    )


@router.post(
    "/api/v1/simplify",
    response_model=SimplificationResponse,
    summary="Generate three medically faithful report versions",
    description=(
        "Runs the approved immutable local Qwen3 checkpoint once and returns Clinical, "
        "General Public, and Child-Friendly versions. Outputs are source-grounded and "
        "require clinical review; the service does not provide diagnosis or treatment advice."
    ),
    responses={
        200: {"description": "All three validated simplification levels were generated."},
        422: {"description": "Input was invalid or generated output failed grounding checks."},
        503: {"description": "The approved pinned local Qwen3 runtime is unavailable."},
    },
)
async def simplify(
    payload: SimplificationRequest,
    request: Request,
    service: Dependency,
) -> SimplificationResponse:
    result = await _process(payload, request, service)
    metadata = service.provider.metadata()
    levels = {value.result.level: _level_schema(value, result, service) for value in result.levels}
    return SimplificationResponse(
        request_id=result.request_id,
        clinical=levels["clinical"],
        general_public=levels["general_public"],
        child_friendly=levels["child_friendly"],
        processing_time_ms=result.processing_time_ms,
        inference=InferenceMetadataSchema(
            provider_name=metadata.provider_name,
            model_name=metadata.model_name,
            model_revision=metadata.model_revision,
            device=metadata.device,
            prompt_version=metadata.prompt_version,
            prompt_tokens=result.provider_result.prompt_tokens,
            output_tokens=result.provider_result.output_tokens,
            generation_time_ms=result.provider_result.generation_time_ms,
            deterministic_generation=True,
            local_files_only=True,
        ),
    )


@router.post(
    "/api/v1/simplifications",
    response_model=LegacySimplificationResponse,
    deprecated=True,
    summary="Simplify a report using the legacy MVP response",
    description=(
        "Compatibility operation. It executes the production three-level provider and "
        "maps the Clinical result into the original MVP response shape."
    ),
    responses={
        200: {"description": "Clinical-level simplification returned."},
        422: {"description": "Input or generated output failed validation."},
        503: {"description": "Approved local Qwen3 runtime unavailable."},
    },
)
async def legacy_simplify(
    payload: SimplificationRequest,
    request: Request,
    service: Dependency,
) -> LegacySimplificationResponse:
    result = await _process(payload, request, service)
    clinical = next(item for item in result.levels if item.result.level == "clinical")
    value = clinical.result
    metadata = service.provider.metadata()
    return LegacySimplificationResponse(
        request_id=result.request_id,
        simplified_report=value.simplified_report,
        sections=LegacySimplificationSectionsSchema(
            executive_summary=value.simplified_report,
            important_findings=list(value.important_findings),
            timeline=[],
            medical_terms_explained=[
                f"{item.term}: {item.explanation}" for item in value.medical_terms_explained
            ],
            simple_explanation=value.simplified_report,
            recommended_follow_up=list(value.suggested_questions_for_doctor),
        ),
        model_name=metadata.model_name,
        model_version=metadata.model_revision,
        provider_name=metadata.provider_name,
        prompt_version=metadata.prompt_version,
        confidence=clinical.confidence,
        processing_time_ms=result.processing_time_ms,
        review_required=clinical.review_required,
        warnings=list(clinical.warnings),
    )


def _status(service: SimplificationService) -> StageModelResponse:
    metadata = service.provider.metadata()
    return StageModelResponse(
        status="HEALTHY" if metadata.ready else "UNAVAILABLE",
        provider_name=metadata.provider_name,
        model_name=metadata.model_name,
        model_revision=metadata.model_revision,
        prompt_version=metadata.prompt_version,
        device=metadata.device,
        detail=metadata.detail,
        model_loading_time_ms=metadata.model_loading_time_ms,
        configuration=metadata.configuration,
    )


@router.get(
    "/api/v1/simplify/health",
    response_model=StageModelResponse,
    summary="Check Qwen3 simplification health",
    description="Reports readiness of the pinned local model and versioned prompt resource.",
    responses={200: {"description": "Provider is healthy."}, 503: {"description": "Unavailable."}},
)
async def health(response: Response, service: Dependency) -> StageModelResponse:
    result = _status(service)
    if result.status != "HEALTHY":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result


@router.get(
    "/api/v1/simplify/models",
    response_model=StageModelResponse,
    summary="Get the approved simplification model",
    description="Returns immutable model, prompt, device, and local-only runtime metadata.",
)
async def models(service: Dependency) -> StageModelResponse:
    return _status(service)


@router.get(
    "/api/v1/simplifications/health",
    response_model=StageModelResponse,
    deprecated=True,
    include_in_schema=False,
)
async def legacy_health(response: Response, service: Dependency) -> StageModelResponse:
    return await health(response, service)


@router.get(
    "/api/v1/simplifications/models",
    response_model=StageModelResponse,
    deprecated=True,
    include_in_schema=False,
)
async def legacy_models(service: Dependency) -> StageModelResponse:
    return _status(service)

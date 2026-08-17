"""Medical verification HTTP endpoints."""
# ruff: noqa: E501

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status

from app.verification.schemas import (
    VerificationModelResponse,
    VerificationRequest,
    VerificationResponse,
)
from app.verification.service import MedicalVerificationService

router = APIRouter(prefix="/api/v1/verification", tags=["Medical Verification"])


def get_service(request: Request) -> MedicalVerificationService:
    return request.app.state.verification_service


Dependency = Annotated[MedicalVerificationService, Depends(get_service)]


@router.post(
    "",
    response_model=VerificationResponse,
    summary="Verify a simplified clinical statement",
    description="Combines deterministic clinical-value checks with local PubMedBERT MedNLI.",
    responses={200: {"description": "Verification completed."}, 503: {"description": "Model unavailable."}},
)
async def verify(payload: VerificationRequest, request: Request, service: Dependency) -> VerificationResponse:
    result = await service.verify(payload.premise, payload.hypothesis, request.state.request_id)
    return VerificationResponse(
        request_id=result.request_id,
        label=result.label,
        probabilities=result.probabilities,
        verification=result.status,
        review_required=result.review_required,
        deterministic_mismatches=list(result.deterministic_mismatches),
        processing_time_ms=result.processing_time_ms,
        nli_latency_ms=result.nli_latency_ms,
        model_name=result.model_name,
        model_revision=result.model_revision,
    )


def _model(service: MedicalVerificationService) -> VerificationModelResponse:
    item = service.provider.metadata()
    return VerificationModelResponse(
        status="ready" if item.ready else "unavailable",
        model_name=item.model_name,
        model_revision=item.model_revision,
        device=item.device,
        detail=item.detail,
        configuration=item.configuration,
    )


@router.get("/health", response_model=VerificationModelResponse, summary="Check verification health")
async def health(response: Response, service: Dependency) -> VerificationModelResponse:
    result = _model(service)
    if result.status != "ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result


@router.get("/models", response_model=VerificationModelResponse, summary="Get verification model metadata")
async def models(service: Dependency) -> VerificationModelResponse:
    return _model(service)

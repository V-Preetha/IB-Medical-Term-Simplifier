"""IndicTrans2 translation HTTP API."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status

from app.translation.provider import SUPPORTED_LANGUAGES
from app.translation.schemas import (
    TranslationModelResponse,
    TranslationRequest,
    TranslationResponse,
)
from app.translation.service import TranslationService

router = APIRouter(prefix="/api/v1/translations", tags=["Translation"])


def get_service(request: Request) -> TranslationService:
    return request.app.state.translation_service


Dependency = Annotated[TranslationService, Depends(get_service)]


@router.post(
    "",
    response_model=TranslationResponse,
    summary="Translate a simplified report",
    description=(
        "Translates the patient-friendly English report with a local IndicTrans2 "
        "provider and fails closed if protected numeric medical content is not preserved."
    ),
    responses={
        200: {"description": "Report translated."},
        422: {"description": "Invalid input or preservation failure."},
        503: {"description": "Approved IndicTrans2 runtime unavailable."},
    },
)
async def translate(
    payload: TranslationRequest,
    request: Request,
    service: Dependency,
) -> TranslationResponse:
    result = await service.process(
        payload.text,
        payload.source_language,
        payload.target_language,
        request.state.request_id,
    )
    metadata = service.provider.metadata()
    return TranslationResponse(
        request_id=result.request_id,
        source_language=payload.source_language,
        target_language=payload.target_language,
        translated_text=result.translated_text,
        provider_name=metadata.provider_name,
        model_name=metadata.model_name,
        model_version=metadata.model_revision,
        processing_time_ms=result.processing_time_ms,
    )


def _status(service: TranslationService) -> TranslationModelResponse:
    metadata = service.provider.metadata()
    return TranslationModelResponse(
        status="ready" if metadata.ready else "not_configured",
        provider_name=metadata.provider_name,
        model_name=metadata.model_name,
        model_revision=metadata.model_revision,
        device=metadata.device,
        detail=metadata.detail,
        supported_languages=SUPPORTED_LANGUAGES,
        configuration=metadata.configuration,
    )


@router.get(
    "/health",
    response_model=TranslationModelResponse,
    summary="Check translation readiness",
    description="Checks the configured local IndicTrans2 runtime without inference.",
)
async def health(response: Response, service: Dependency) -> TranslationModelResponse:
    result = _status(service)
    if result.status != "ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result


@router.get(
    "/models",
    response_model=TranslationModelResponse,
    summary="Get translation model metadata",
    description="Returns IndicTrans2 identity and the supported MVP language registry.",
)
async def models(service: Dependency) -> TranslationModelResponse:
    return _status(service)

"""FastAPI dependency adapters for the OCR application boundary."""

from typing import Annotated
from uuid import UUID

from fastapi import Depends, Header, Request

from app.config import settings
from app.ocr.application.service import OCRApplicationService
from app.ocr.observability.logging import RecentLogStore
from app.ocr.providers.contracts import (
    BaseOCRProvider,
    BasePostProcessor,
)
from app.ocr.providers.errors import ProviderUnavailableError
from app.ocr.providers.runtime import ProviderContainer


def get_provider_container(request: Request) -> ProviderContainer:
    container = getattr(request.app.state, "ocr_provider_container", None)
    if not isinstance(container, ProviderContainer):
        raise ProviderUnavailableError("OCR providers are not initialized.")
    return container


ProviderContainerDependency = Annotated[ProviderContainer, Depends(get_provider_container)]


def get_ocr_provider(container: ProviderContainerDependency) -> BaseOCRProvider:
    return container.ocr_provider


def get_postprocessor(container: ProviderContainerDependency) -> BasePostProcessor:
    return container.postprocessor


def get_ocr_application_service(request: Request) -> OCRApplicationService:
    service = getattr(request.app.state, "ocr_application_service", None)
    if not isinstance(service, OCRApplicationService):
        raise ProviderUnavailableError("OCR application service is not initialized.")
    return service


def get_log_store(request: Request) -> RecentLogStore:
    store = getattr(request.app.state, "ocr_log_store", None)
    if not isinstance(store, RecentLogStore):
        raise ProviderUnavailableError("OCR log inspection is not initialized.")
    return store


def get_owner_id(
    x_owner_id: Annotated[UUID | None, Header(alias="X-Owner-ID")] = None,
) -> UUID:
    return x_owner_id or settings.default_owner_id


OCRApplicationServiceDependency = Annotated[
    OCRApplicationService, Depends(get_ocr_application_service)
]
OwnerDependency = Annotated[UUID, Depends(get_owner_id)]
LogStoreDependency = Annotated[RecentLogStore, Depends(get_log_store)]
OCRProviderDependency = Annotated[BaseOCRProvider, Depends(get_ocr_provider)]
PostProcessorDependency = Annotated[BasePostProcessor, Depends(get_postprocessor)]

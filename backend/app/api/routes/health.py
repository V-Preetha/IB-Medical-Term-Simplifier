"""Health and metadata routes."""

import logging

from fastapi import APIRouter, Depends

from app.config.settings import Settings, get_settings
from app.schemas.health import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Return service health and configuration metadata safe for clients."""
    logger.debug("Health check requested")
    return HealthResponse(
        status="ok",
        service_name=settings.app_name,
        environment=settings.environment,
        version=settings.app_version,
    )


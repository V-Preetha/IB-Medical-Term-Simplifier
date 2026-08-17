"""Provider-neutral entity-linking application service."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from time import perf_counter
from uuid import UUID, uuid4

from app.entity_linking.contracts import (
    BaseEntityLinkingProvider,
    EntityLink,
    LinkerHealth,
    SourceEntity,
)
from app.entity_linking.errors import EntityLinkingError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EntityLinkingResult:
    request_id: UUID
    links: tuple[EntityLink, ...]
    processing_time_ms: float
    warnings: tuple[str, ...]
    metadata: dict[str, object]


class EntityLinkingService:
    def __init__(self, provider: BaseEntityLinkingProvider) -> None:
        self._provider = provider

    async def initialize(self, *, strict: bool = True) -> None:
        try:
            await self._provider.initialize()
        except EntityLinkingError:
            if strict:
                raise
            logger.warning(
                "Entity linking is not production-ready",
                extra={
                    "event": "entity_linker_not_ready",
                    "pipeline_stage": "entity_linking_startup",
                },
            )

    async def process(
        self, entities: tuple[SourceEntity, ...], request_id: UUID | None = None
    ) -> EntityLinkingResult:
        trace_id = request_id or uuid4()
        started = perf_counter()
        output = await asyncio.to_thread(self._provider.link, entities)
        elapsed = round((perf_counter() - started) * 1000, 3)
        metadata = self._provider.metadata()
        linked = sum(link.selected_concept is not None for link in output.links)
        logger.info(
            "Entity linking completed",
            extra={
                "event": "entity_linking_completed",
                "request_id": str(trace_id),
                "pipeline_stage": "entity_linking",
                "provider_name": metadata.provider_name,
                "provider_version": metadata.provider_version,
                "terminology_version": metadata.terminology_version,
                "processing_time_ms": elapsed,
                "entity_count": len(entities),
                "linked_entity_count": linked,
            },
        )
        return EntityLinkingResult(
            request_id=trace_id,
            links=output.links,
            processing_time_ms=elapsed,
            warnings=output.warnings,
            metadata={
                "provider_name": metadata.provider_name,
                "provider_version": metadata.provider_version,
                "model_name": metadata.model_name,
                "model_version": metadata.model_version,
                "terminology_name": metadata.terminology_name,
                "terminology_version": metadata.terminology_version,
                "confidence_method": metadata.confidence_method,
                "calibration_version": metadata.calibration_version,
                "loading_time_ms": metadata.loading_time_ms,
                "startup_timestamp": metadata.startup_timestamp,
                "configuration": metadata.configuration,
            },
        )

    def health(self) -> LinkerHealth:
        return self._provider.health()

    async def shutdown(self) -> None:
        await self._provider.shutdown()

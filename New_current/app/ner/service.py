"""Production medical NER application service."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from time import perf_counter
from uuid import UUID, uuid4

from app.ner.contracts import BaseNERProvider, NERProviderHealth, NormalizedEntity

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MedicalNERResult:
    request_id: UUID
    provider_name: str
    model_name: str
    model_revision: str
    entities: tuple[NormalizedEntity, ...]
    confidence: float | None
    processing_time_ms: float
    token_count: int | None
    tokens_per_second: float | None
    device: str
    warnings: tuple[str, ...]
    metadata: dict[str, object]


class MedicalNERService:
    """Run the approved model through its provider-neutral contract."""

    def __init__(self, provider: BaseNERProvider) -> None:
        self._provider = provider

    async def initialize(self) -> None:
        await self._provider.initialize()

    async def process(self, text: str, request_id: UUID | None = None) -> MedicalNERResult:
        trace_id = request_id or uuid4()
        started = perf_counter()
        output = await asyncio.to_thread(self._provider.extract, text)
        processing_time_ms = round((perf_counter() - started) * 1000, 3)
        metadata = self._provider.metadata()
        confidence = (
            round(sum(entity.confidence for entity in output.entities) / len(output.entities), 6)
            if output.entities
            else None
        )
        tokens_per_second = (
            round(output.token_count / (processing_time_ms / 1000), 3)
            if output.token_count is not None and processing_time_ms > 0
            else None
        )
        logger.info(
            "Production medical NER inference completed",
            extra={
                "event": "ner_inference_completed",
                "request_id": str(trace_id),
                "pipeline_stage": "medical_ner",
                "provider_name": metadata.provider_name,
                "model_name": metadata.model_name,
                "model_revision": metadata.model_revision,
                "processing_time_ms": processing_time_ms,
                "entity_count": len(output.entities),
                "confidence": confidence,
                "device": metadata.device,
            },
        )
        return MedicalNERResult(
            request_id=trace_id,
            provider_name=metadata.provider_name,
            model_name=metadata.model_name,
            model_revision=metadata.model_revision,
            entities=output.entities,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            token_count=output.token_count,
            tokens_per_second=tokens_per_second,
            device=metadata.device,
            warnings=output.warnings,
            metadata={
                "framework": metadata.framework,
                "loading_time_ms": metadata.loading_time_ms,
                "startup_timestamp": metadata.startup_timestamp,
                "configuration": metadata.configuration,
            },
        )

    def health(self) -> NERProviderHealth:
        return self._provider.health()

    async def shutdown(self) -> None:
        await self._provider.shutdown()

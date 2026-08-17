"""Application service for provider-neutral relation extraction."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from time import perf_counter
from uuid import UUID, uuid4

from app.relation_extraction.contracts import (
    BaseRelationExtractionProvider,
    ClinicalRelation,
    RelationDocument,
    RelationProviderHealth,
)
from app.relation_extraction.errors import RelationExtractionError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RelationExtractionResult:
    request_id: UUID
    relations: tuple[ClinicalRelation, ...]
    processing_time_ms: float
    candidate_pair_count: int
    token_count: int
    tokens_per_second: float | None
    warnings: tuple[str, ...]
    metadata: dict[str, object]


class RelationExtractionService:
    def __init__(self, provider: BaseRelationExtractionProvider) -> None:
        self._provider = provider

    async def initialize(self, *, strict: bool = True) -> None:
        try:
            await self._provider.initialize()
        except RelationExtractionError:
            if strict:
                raise
            logger.warning(
                "Relation extraction is not production-ready",
                extra={
                    "event": "relation_provider_not_ready",
                    "pipeline_stage": "relation_extraction_startup",
                },
            )

    async def process(
        self, document: RelationDocument, request_id: UUID | None = None
    ) -> RelationExtractionResult:
        trace_id = request_id or uuid4()
        started = perf_counter()
        output = await asyncio.to_thread(self._provider.extract, document)
        elapsed = round((perf_counter() - started) * 1_000, 3)
        metadata = self._provider.metadata()
        throughput = (
            round(output.token_count / (elapsed / 1_000), 3)
            if output.token_count and elapsed > 0
            else None
        )
        average_confidence = (
            round(
                sum(relation.confidence for relation in output.relations) / len(output.relations),
                6,
            )
            if output.relations
            else None
        )
        logger.info(
            "Relation extraction completed",
            extra={
                "event": "relation_extraction_completed",
                "request_id": str(trace_id),
                "pipeline_stage": "relation_extraction",
                "provider_name": metadata.provider_name,
                "model_name": metadata.model_name,
                "model_revision": metadata.model_revision,
                "processing_time_ms": elapsed,
                "candidate_pair_count": output.candidate_pair_count,
                "relation_count": len(output.relations),
                "confidence": average_confidence,
                "device": metadata.device,
            },
        )
        return RelationExtractionResult(
            request_id=trace_id,
            relations=output.relations,
            processing_time_ms=elapsed,
            candidate_pair_count=output.candidate_pair_count,
            token_count=output.token_count,
            tokens_per_second=throughput,
            warnings=output.warnings,
            metadata={
                "provider_name": metadata.provider_name,
                "provider_version": metadata.provider_version,
                "model_name": metadata.model_name,
                "model_revision": metadata.model_revision,
                "framework": metadata.framework,
                "device": metadata.device,
                "relation_labels": metadata.relation_labels,
                "confidence_method": metadata.confidence_method,
                "calibration_version": metadata.calibration_version,
                "preprocessing_version": metadata.preprocessing_version,
                "startup_timestamp": metadata.startup_timestamp,
                "loading_time_ms": metadata.loading_time_ms,
                "configuration": metadata.configuration,
            },
        )

    def health(self) -> RelationProviderHealth:
        return self._provider.health()

    async def shutdown(self) -> None:
        await self._provider.shutdown()

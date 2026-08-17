"""Provider-neutral medical embedding application service."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from time import perf_counter
from uuid import UUID, uuid4

from app.embeddings.contracts import (
    BaseEmbeddingProvider,
    EmbeddingInput,
    EmbeddingProviderHealth,
    EmbeddingVector,
)
from app.embeddings.errors import EmbeddingError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MedicalEmbeddingResult:
    request_id: UUID
    embeddings: tuple[EmbeddingVector, ...]
    processing_time_ms: float
    tokens_per_second: float | None
    warnings: tuple[str, ...]
    metadata: dict[str, object]


class MedicalEmbeddingService:
    def __init__(self, provider: BaseEmbeddingProvider) -> None:
        self._provider = provider

    async def initialize(self, *, strict: bool = True) -> None:
        try:
            await self._provider.initialize()
        except EmbeddingError:
            if strict:
                raise
            logger.warning(
                "Medical embeddings are not production-ready",
                extra={
                    "event": "embedding_provider_not_ready",
                    "pipeline_stage": "embedding_startup",
                },
            )

    async def process(
        self,
        inputs: tuple[EmbeddingInput, ...],
        request_id: UUID | None = None,
    ) -> MedicalEmbeddingResult:
        trace_id = request_id or uuid4()
        started = perf_counter()
        output = await asyncio.to_thread(self._provider.embed_batch, inputs)
        elapsed = round((perf_counter() - started) * 1_000, 3)
        metadata = self._provider.metadata()
        token_count = sum(item.token_count for item in output.embeddings)
        throughput = (
            round(token_count / (elapsed / 1_000), 3) if token_count and elapsed > 0 else None
        )
        logger.info(
            "Medical embedding batch completed",
            extra={
                "event": "embedding_batch_completed",
                "request_id": str(trace_id),
                "pipeline_stage": "medical_embeddings",
                "provider_name": metadata.provider_name,
                "model_name": metadata.model_name,
                "model_revision": metadata.model_revision,
                "processing_time_ms": elapsed,
                "batch_size": len(inputs),
                "token_count": token_count,
                "dimensions": metadata.dimensions,
                "device": metadata.device,
            },
        )
        return MedicalEmbeddingResult(
            request_id=trace_id,
            embeddings=output.embeddings,
            processing_time_ms=elapsed,
            tokens_per_second=throughput,
            warnings=output.warnings,
            metadata={
                "provider_name": metadata.provider_name,
                "provider_version": metadata.provider_version,
                "model_name": metadata.model_name,
                "model_revision": metadata.model_revision,
                "framework": metadata.framework,
                "device": metadata.device,
                "dimensions": metadata.dimensions,
                "pooling_method": metadata.pooling_method,
                "normalized": metadata.normalized,
                "startup_timestamp": metadata.startup_timestamp,
                "loading_time_ms": metadata.loading_time_ms,
                "configuration": metadata.configuration,
            },
        )

    def health(self) -> EmbeddingProviderHealth:
        return self._provider.health()

    async def shutdown(self) -> None:
        await self._provider.shutdown()

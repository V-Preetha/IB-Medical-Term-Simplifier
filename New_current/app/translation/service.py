"""Application service for IndicTrans2 translation."""

import asyncio
import logging
from dataclasses import dataclass
from time import perf_counter
from uuid import UUID

from app.translation.contracts import BaseTranslationProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TranslationResult:
    request_id: UUID
    translated_text: str
    processing_time_ms: float


@dataclass(frozen=True, slots=True)
class BatchTranslationResult:
    request_id: UUID
    translated_texts: tuple[str, ...]
    processing_time_ms: float


class TranslationService:
    def __init__(self, provider: BaseTranslationProvider) -> None:
        self.provider = provider

    async def initialize(self, *, strict: bool = True) -> None:
        await self.provider.initialize(strict=strict)

    async def process(
        self,
        text: str,
        source_language: str,
        target_language: str,
        request_id: UUID,
    ) -> TranslationResult:
        batch = await self.process_batch((text,), source_language, target_language, request_id)
        return TranslationResult(request_id, batch.translated_texts[0], batch.processing_time_ms)

    async def process_batch(
        self,
        texts: tuple[str, ...],
        source_language: str,
        target_language: str,
        request_id: UUID,
    ) -> BatchTranslationResult:
        """Translate several texts (e.g. all three simplification levels) in one call.

        Amortizes model-invocation overhead across texts instead of one sequential
        call per text.
        """

        started = perf_counter()
        outputs = await asyncio.to_thread(
            self.provider.translate_batch, texts, source_language, target_language
        )
        elapsed = round((perf_counter() - started) * 1000, 3)
        metadata = self.provider.metadata()
        logger.info(
            "IndicTrans2 translation completed",
            extra={
                "event": "translation_completed",
                "request_id": str(request_id),
                "pipeline_stage": "translation",
                "model_name": metadata.model_name,
                "model_revision": metadata.model_revision,
                "processing_time_ms": elapsed,
                "target_language": target_language,
                "batch_size": len(texts),
            },
        )
        return BatchTranslationResult(request_id, outputs, elapsed)

    async def shutdown(self) -> None:
        await self.provider.shutdown()

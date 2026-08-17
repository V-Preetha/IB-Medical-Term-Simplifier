"""Application service and source-grounding controls for Qwen3 simplification."""

import asyncio
import logging
import re
from dataclasses import dataclass
from time import perf_counter
from uuid import UUID

from app.simplification.contracts import (
    BaseSimplificationProvider,
    ProviderSimplificationResult,
    SimplificationLevelResult,
)
from app.simplification.errors import SimplificationOutputError

logger = logging.getLogger(__name__)
_FACT = re.compile(
    r"(?<![\w.])\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?\s*"
    r"(?:%|mg/dL|g/dL|mmol/L|mEq/L|U/L|IU/L|mmHg|bpm|mg|mcg|mL|mm|cm)?",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ValidatedLevel:
    result: SimplificationLevelResult
    confidence: float | None
    warnings: tuple[str, ...]
    review_required: bool


@dataclass(frozen=True, slots=True)
class SimplificationResult:
    request_id: UUID
    original_report: str
    levels: tuple[ValidatedLevel, ...]
    processing_time_ms: float
    provider_result: ProviderSimplificationResult


class SimplificationService:
    def __init__(self, provider: BaseSimplificationProvider) -> None:
        self.provider = provider

    async def initialize(self, *, strict: bool = True) -> None:
        await self.provider.initialize(strict=strict)

    async def process(
        self,
        text: str,
        entities: dict[str, tuple[str, ...]],
        request_id: UUID,
        linked_concepts: tuple[dict[str, str], ...] = (),
    ) -> SimplificationResult:
        started = perf_counter()
        generated = await asyncio.to_thread(
            self.provider.simplify, text, entities, linked_concepts
        )
        levels = tuple(self._validate_level(text, entities, level) for level in generated.levels)
        elapsed = round((perf_counter() - started) * 1000, 3)
        metadata = self.provider.metadata()
        logger.info(
            "Qwen3 medical report simplification completed",
            extra={
                "event": "simplification_completed",
                "request_id": str(request_id),
                "pipeline_stage": "simplification",
                "model_name": metadata.model_name,
                "model_revision": metadata.model_revision,
                "prompt_version": metadata.prompt_version,
                "processing_time_ms": elapsed,
                "generation_time_ms": generated.generation_time_ms,
                "prompt_tokens": generated.prompt_tokens,
                "output_tokens": generated.output_tokens,
                "review_required": any(level.review_required for level in levels),
            },
        )
        return SimplificationResult(request_id, text, levels, elapsed, generated)

    @classmethod
    def _validate_level(
        cls,
        source: str,
        entities: dict[str, tuple[str, ...]],
        level: SimplificationLevelResult,
    ) -> ValidatedLevel:
        rendered = " ".join(
            (
                level.simplified_report,
                *level.important_findings,
                *(item.term for item in level.medical_terms_explained),
                *(item.explanation for item in level.medical_terms_explained),
            )
        )
        source_facts = cls._facts(source)
        output_facts = cls._facts(rendered)
        introduced = output_facts - source_facts
        if introduced:
            raise SimplificationOutputError(
                "Simplification introduced a numeric value or unit absent from the source."
            )
        missing = source_facts - output_facts
        source_folded = source.casefold()
        allowed_terms = {
            value.casefold() for values in entities.values() for value in values if value.strip()
        }
        for item in level.medical_terms_explained:
            if (
                item.term.casefold() not in source_folded
                and item.term.casefold() not in allowed_terms
            ):
                raise SimplificationOutputError(
                    "Simplification explained a medical term absent from the source evidence."
                )
        entity_terms = tuple(
            dict.fromkeys(value.casefold() for values in entities.values() for value in values)
        )
        entity_hits = sum(term in rendered.casefold() for term in entity_terms)
        components: list[float] = []
        if source_facts:
            components.append((len(source_facts) - len(missing)) / len(source_facts))
        if entity_terms:
            components.append(entity_hits / len(entity_terms))
        confidence = round(sum(components) / len(components), 6) if components else None
        warnings: list[str] = []
        if missing:
            warnings.append("One or more source numeric facts were not repeated in this version.")
        if entity_terms and entity_hits < len(entity_terms):
            warnings.append(
                "One or more supplied medical entities were not repeated in this version."
            )
        warnings.append(
            "Confidence is a deterministic source-fact preservation score, "
            "not a calibrated clinical probability."
        )
        return ValidatedLevel(level, confidence, tuple(warnings), True)

    @staticmethod
    def _facts(text: str) -> set[str]:
        return {"".join(match.group(0).casefold().split()) for match in _FACT.finditer(text)}

    async def shutdown(self) -> None:
        await self.provider.shutdown()

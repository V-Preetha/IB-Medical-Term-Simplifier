"""SciSpaCy UMLS adapter and provider registry."""

from __future__ import annotations

import asyncio
import gc
import importlib.metadata
import logging
import os
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from threading import RLock
from time import perf_counter
from typing import Any

from app.entity_linking.config import EntityLinkingSettings
from app.entity_linking.contracts import (
    BaseEntityLinkingProvider,
    ConceptCandidate,
    EntityLink,
    LinkerHealth,
    LinkerHealthStatus,
    LinkerMetadata,
    LinkStatus,
    ProviderLinkResult,
    SourceEntity,
)
from app.entity_linking.errors import (
    EntityLinkingConfigurationError,
    EntityLinkingInferenceError,
    EntityLinkingUnavailableError,
    UnsupportedEntityLinkingProviderError,
)

logger = logging.getLogger(__name__)
ProviderFactory = Callable[[], BaseEntityLinkingProvider]


class EntityLinkingProviderRegistry:
    """Instance-scoped registry used by the application composition root."""

    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, name: str, factory: ProviderFactory) -> None:
        key = name.strip().casefold()
        if not key or key in self._factories:
            raise EntityLinkingConfigurationError(
                f"Entity-linking provider registration is invalid: {name!r}."
            )
        self._factories[key] = factory
        logger.info(
            "Entity-linking provider registered",
            extra={"event": "entity_linker_registered", "provider_name": key},
        )

    def create(self, name: str) -> BaseEntityLinkingProvider:
        key = name.strip().casefold()
        factory = self._factories.get(key)
        if factory is None:
            raise UnsupportedEntityLinkingProviderError(
                f"Unsupported entity-linking provider: {name}."
            )
        return factory()


class SciSpacyUMLSProvider(BaseEntityLinkingProvider):
    """Local-only SciSpaCy adapter for a licensed UMLS knowledge base."""

    def __init__(self, settings: EntityLinkingSettings) -> None:
        self._settings = settings
        self._nlp: Any = None
        self._linker: Any = None
        self._startup_timestamp: datetime | None = None
        self._loading_time_ms: float | None = None
        self._health_detail = "Provider has not been initialized."
        self._lock = RLock()

    async def initialize(self) -> None:
        if self._linker is not None:
            return
        self._settings.validate()
        started = perf_counter()
        try:
            await asyncio.to_thread(self._load_runtime)
        except EntityLinkingConfigurationError:
            raise
        except Exception as exc:
            self._health_detail = "SciSpaCy UMLS initialization failed."
            raise EntityLinkingUnavailableError(
                "SciSpaCy UMLS could not initialize from approved local artifacts."
            ) from exc
        self._loading_time_ms = round((perf_counter() - started) * 1000, 3)
        self._startup_timestamp = datetime.now(UTC)
        self._health_detail = "SciSpaCy UMLS is initialized from local artifacts."
        logger.info(
            "Entity-linking provider initialized",
            extra={
                "event": "entity_linker_initialized",
                "provider_name": self._settings.provider,
                "provider_version": self._settings.provider_version,
                "terminology_version": self._settings.terminology_version,
                "model_loading_time_ms": self._loading_time_ms,
            },
        )

    def link(self, entities: tuple[SourceEntity, ...]) -> ProviderLinkResult:
        if self._linker is None:
            raise EntityLinkingUnavailableError("SciSpaCy UMLS is not initialized.")
        try:
            with self._lock:
                raw_candidates = self._linker.candidate_generator(
                    [entity.text for entity in entities],
                    k=self._settings.max_candidates,
                )
            links = tuple(
                self._normalize_link(entity, candidates)
                for entity, candidates in zip(entities, raw_candidates, strict=True)
            )
        except EntityLinkingInferenceError:
            raise
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise EntityLinkingInferenceError(
                "SciSpaCy UMLS returned invalid candidate data."
            ) from exc
        return ProviderLinkResult(links=links)

    def metadata(self) -> LinkerMetadata:
        return LinkerMetadata(
            provider_name=self._settings.provider,
            provider_version=self._settings.provider_version,
            model_name=self._settings.language_model,
            model_version=self._settings.language_model_version,
            terminology_name=self._settings.terminology,
            terminology_version=self._settings.terminology_version,
            confidence_method="scispacy_candidate_similarity",
            calibration_version="uncalibrated-scispacy-umls-v1",
            startup_timestamp=self._startup_timestamp,
            loading_time_ms=self._loading_time_ms,
            configuration={
                "local_files_only": True,
                "language_model_path": str(self._settings.language_model_path or ""),
                "knowledge_base_path": str(self._settings.knowledge_base_path or ""),
                "confidence_threshold": self._settings.confidence_threshold,
                "max_candidates": self._settings.max_candidates,
                "ambiguity_delta": self._settings.ambiguity_delta,
                "license_accepted": self._settings.license_accepted,
            },
        )

    def health(self) -> LinkerHealth:
        if self._linker is not None:
            status = LinkerHealthStatus.READY
        else:
            try:
                self._settings.validate()
            except EntityLinkingConfigurationError as exc:
                status = LinkerHealthStatus.NOT_CONFIGURED
                detail = exc.message
            else:
                status = LinkerHealthStatus.NOT_INITIALIZED
                detail = self._health_detail
            return LinkerHealth(status, detail, self.metadata())
        return LinkerHealth(status, self._health_detail, self.metadata())

    async def shutdown(self) -> None:
        was_initialized = self._linker is not None
        self._linker = None
        self._nlp = None
        gc.collect()
        self._health_detail = "Provider has been shut down."
        if was_initialized:
            logger.info(
                "Entity-linking provider shut down",
                extra={
                    "event": "entity_linker_shutdown",
                    "provider_name": self._settings.provider,
                },
            )

    def _load_runtime(self) -> None:
        installed = importlib.metadata.version("scispacy")
        if installed != self._settings.provider_version:
            raise EntityLinkingConfigurationError(
                "Installed SciSpaCy version does not match the approved manifest."
            )
        os.environ["SCISPACY_CACHE"] = str(self._settings.knowledge_base_path)
        import scispacy  # noqa: F401
        import spacy

        self._nlp = spacy.load(str(self._settings.language_model_path))
        self._nlp.add_pipe(
            "scispacy_linker",
            config={
                "linker_name": "umls",
                "resolve_abbreviations": False,
                "threshold": self._settings.confidence_threshold,
                "max_entities_per_mention": self._settings.max_candidates,
            },
        )
        self._linker = self._nlp.get_pipe("scispacy_linker")

    def _normalize_link(self, entity: SourceEntity, raw_candidates: Iterable[Any]) -> EntityLink:
        candidates: list[ConceptCandidate] = []
        for candidate in raw_candidates:
            concept_id = str(candidate.concept_id)
            similarities = tuple(float(value) for value in candidate.similarities)
            if not similarities:
                continue
            confidence = max(similarities)
            if confidence < self._settings.confidence_threshold:
                continue
            concept = self._linker.kb.cui_to_entity[concept_id]
            candidates.append(
                ConceptCandidate(
                    concept_id=concept_id,
                    preferred_name=str(concept.canonical_name),
                    semantic_types=tuple(str(value) for value in concept.types),
                    confidence=round(confidence, 6),
                    source_ontology=self._settings.terminology,
                )
            )
        ranked = tuple(
            sorted(candidates, key=lambda item: item.confidence, reverse=True)[
                : self._settings.max_candidates
            ]
        )
        if not ranked:
            return EntityLink(entity, LinkStatus.UNLINKED, None, (), True)
        ambiguous = (
            len(ranked) > 1
            and ranked[0].confidence - ranked[1].confidence <= self._settings.ambiguity_delta
        )
        return EntityLink(
            entity,
            LinkStatus.AMBIGUOUS if ambiguous else LinkStatus.LINKED,
            ranked[0],
            ranked,
            ambiguous,
        )


def create_production_registry(
    settings: EntityLinkingSettings,
) -> EntityLinkingProviderRegistry:
    registry = EntityLinkingProviderRegistry()
    registry.register("scispacy-umls", lambda: SciSpacyUMLSProvider(settings))
    return registry

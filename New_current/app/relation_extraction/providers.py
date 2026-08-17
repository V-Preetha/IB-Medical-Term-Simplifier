"""Local BioLinkBERT relation-classification adapter and registry."""

from __future__ import annotations

import asyncio
import gc
import json
import logging
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from threading import RLock
from time import perf_counter
from typing import Any

from app.relation_extraction.config import RelationSettings
from app.relation_extraction.contracts import (
    BaseRelationExtractionProvider,
    ClinicalRelation,
    ProviderRelationResult,
    RelationDocument,
    RelationEntity,
    RelationHealthStatus,
    RelationProviderHealth,
    RelationProviderMetadata,
)
from app.relation_extraction.errors import (
    RelationConfigurationError,
    RelationInferenceError,
    RelationProviderUnavailableError,
    UnsupportedRelationProviderError,
)
from app.relation_extraction.manifest import RelationModelManifest

logger = logging.getLogger(__name__)
ProviderFactory = Callable[[], BaseRelationExtractionProvider]
_GENERIC_LABEL_PREFIX = "LABEL_"


class RelationProviderRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, name: str, factory: ProviderFactory) -> None:
        key = name.strip().casefold()
        if not key or key in self._factories:
            raise RelationConfigurationError(
                f"Relation provider registration is invalid: {name!r}."
            )
        self._factories[key] = factory
        logger.info(
            "Relation provider registered",
            extra={"event": "relation_provider_registered", "provider_name": key},
        )

    def create(self, name: str) -> BaseRelationExtractionProvider:
        factory = self._factories.get(name.strip().casefold())
        if factory is None:
            raise UnsupportedRelationProviderError(f"Unsupported relation provider: {name}.")
        return factory()


class BioLinkBERTRelationProvider(BaseRelationExtractionProvider):
    """Sequence-classification adapter for an approved BioLinkBERT relation head."""

    def __init__(self, settings: RelationSettings, manifest: RelationModelManifest) -> None:
        self._settings = settings
        self._manifest = manifest
        self._tokenizer: Any = None
        self._model: Any = None
        self._device = settings.device
        self._labels: dict[int, str] = {}
        self._no_relation_labels: frozenset[str] = frozenset()
        self._startup_timestamp: datetime | None = None
        self._loading_time_ms: float | None = None
        self._health_detail = "Provider has not been initialized."
        self._lock = RLock()

    async def initialize(self) -> None:
        if self._model is not None:
            return
        self._settings.validate(self._manifest)
        self._validate_checkpoint_contract()
        started = perf_counter()
        try:
            await asyncio.to_thread(self._load_runtime)
        except (RelationConfigurationError, RelationProviderUnavailableError):
            raise
        except Exception as exc:
            self._health_detail = "BioLinkBERT relation provider initialization failed."
            raise RelationProviderUnavailableError(
                "BioLinkBERT could not initialize from the approved local checkpoint."
            ) from exc
        self._loading_time_ms = round((perf_counter() - started) * 1_000, 3)
        self._startup_timestamp = datetime.now(UTC)
        self._health_detail = "BioLinkBERT relation provider is ready."
        logger.info(
            "Relation provider initialized",
            extra={
                "event": "relation_provider_initialized",
                "provider_name": self._manifest.provider,
                "model_name": self._manifest.repository_id,
                "model_revision": self._manifest.pinned_revision,
                "device": self._device,
                "model_loading_time_ms": self._loading_time_ms,
                "relation_label_count": len(self._labels),
            },
        )

    def extract(self, document: RelationDocument) -> ProviderRelationResult:
        if self._model is None:
            raise RelationProviderUnavailableError(
                "BioLinkBERT relation provider is not initialized."
            )
        _validate_document(document)
        pairs, warnings = _candidate_pairs(document.entities, self._settings.max_entity_pairs)
        if not pairs:
            return ProviderRelationResult((), 0, 0, warnings)
        relations: list[ClinicalRelation] = []
        token_count = 0
        try:
            import torch

            for offset in range(0, len(pairs), self._settings.batch_size):
                batch = pairs[offset : offset + self._settings.batch_size]
                marked = [_mark_entities(document.text, source, target) for source, target in batch]
                with self._lock:
                    encoded = self._tokenizer(
                        marked,
                        padding=True,
                        truncation=True,
                        max_length=self._settings.max_length,
                        return_tensors="pt",
                    )
                    token_count += int(encoded["attention_mask"].sum().item())
                    model_inputs = {key: value.to(self._device) for key, value in encoded.items()}
                    with torch.inference_mode():
                        probabilities = torch.softmax(self._model(**model_inputs).logits, dim=-1)
                scores, identifiers = probabilities.max(dim=-1)
                for (source, target), identifier, score in zip(
                    batch,
                    identifiers.detach().cpu().tolist(),
                    scores.detach().cpu().tolist(),
                    strict=True,
                ):
                    label = self._labels[int(identifier)]
                    confidence = float(score)
                    if (
                        label in self._no_relation_labels
                        or confidence < self._settings.confidence_threshold
                    ):
                        continue
                    relations.append(
                        ClinicalRelation(
                            source=source,
                            target=target,
                            relation_type=label,
                            confidence=round(confidence, 6),
                            evidence_start=min(source.start, target.start),
                            evidence_end=max(source.end, target.end),
                        )
                    )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise RelationInferenceError(
                "BioLinkBERT returned invalid relation-classification output."
            ) from exc
        return ProviderRelationResult(tuple(relations), len(pairs), token_count, warnings)

    def metadata(self) -> RelationProviderMetadata:
        return RelationProviderMetadata(
            provider_name=self._manifest.provider,
            provider_version="transformers-sequence-classification-v1",
            model_name=self._manifest.repository_id,
            model_revision=self._manifest.pinned_revision,
            framework=self._manifest.framework,
            device=self._device,
            relation_labels=tuple(self._labels.values()),
            confidence_method="softmax_relation_class_probability",
            calibration_version=self._settings.calibration_version,
            preprocessing_version=self._settings.preprocessing_version,
            startup_timestamp=self._startup_timestamp,
            loading_time_ms=self._loading_time_ms,
            configuration={
                "local_files_only": True,
                "confidence_threshold": self._settings.confidence_threshold,
                "batch_size": self._settings.batch_size,
                "max_length": self._settings.max_length,
                "max_entity_pairs": self._settings.max_entity_pairs,
                "allow_cpu_fallback": self._settings.allow_cpu_fallback,
                "cache_path": str(self._settings.cache_path),
            },
        )

    def health(self) -> RelationProviderHealth:
        if self._model is not None:
            return RelationProviderHealth(
                RelationHealthStatus.READY, self._health_detail, self.metadata()
            )
        try:
            self._settings.validate(self._manifest)
            self._validate_checkpoint_contract()
        except RelationConfigurationError as exc:
            status = (
                RelationHealthStatus.INCOMPATIBLE_ARTIFACT
                if "relation-classification" in exc.message
                else RelationHealthStatus.NOT_CONFIGURED
            )
            return RelationProviderHealth(status, exc.message, self.metadata())
        return RelationProviderHealth(
            RelationHealthStatus.NOT_INITIALIZED,
            self._health_detail,
            self.metadata(),
        )

    async def shutdown(self) -> None:
        was_initialized = self._model is not None
        self._model = None
        self._tokenizer = None
        self._labels = {}
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        if was_initialized:
            logger.info(
                "Relation provider shut down",
                extra={
                    "event": "relation_provider_shutdown",
                    "provider_name": self._manifest.provider,
                    "model_revision": self._manifest.pinned_revision,
                },
            )

    def _validate_checkpoint_contract(self) -> None:
        config_path = self._settings.cache_path / "config.json"
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RelationConfigurationError(
                "The local BioLinkBERT config.json is missing or invalid."
            ) from exc
        architectures = payload.get("architectures", [])
        labels = _normalized_labels(payload.get("id2label"))
        declared_preprocessing = payload.get("relation_preprocessing_version")
        no_relation = self._settings.no_relation_labels or tuple(
            payload.get("no_relation_labels", ())
        )
        if (
            not any("SequenceClassification" in value for value in architectures)
            or len(labels) < 2
            or any(value.startswith(_GENERIC_LABEL_PREFIX) for value in labels.values())
            or declared_preprocessing != self._settings.preprocessing_version
            or not no_relation
        ):
            raise RelationConfigurationError(
                "The approved local BioLinkBERT artifact is not a compatible "
                "relation-classification checkpoint with named labels, no-relation "
                "labels, and the approved preprocessing contract."
            )
        if any(label not in labels.values() for label in no_relation):
            raise RelationConfigurationError(
                "Configured no-relation labels are absent from the model ontology."
            )
        self._labels = labels
        self._no_relation_labels = frozenset(no_relation)

    def _load_runtime(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self._device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self._device == "cuda" and not torch.cuda.is_available():
            if not self._settings.allow_cpu_fallback:
                raise RelationProviderUnavailableError(
                    "CUDA was requested for relation extraction but is unavailable."
                )
            self._device = "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self._settings.cache_path), local_files_only=True
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(self._settings.cache_path), local_files_only=True
        )
        self._model.to(self._device)
        self._model.eval()


def create_production_registry(
    settings: RelationSettings, manifest: RelationModelManifest
) -> RelationProviderRegistry:
    registry = RelationProviderRegistry()
    registry.register(
        manifest.provider,
        lambda: BioLinkBERTRelationProvider(settings, manifest),
    )
    return registry


def _normalized_labels(raw: Any) -> dict[int, str]:
    if not isinstance(raw, Mapping):
        return {}
    try:
        return {
            int(identifier): str(label).strip().upper()
            for identifier, label in raw.items()
            if str(label).strip()
        }
    except (TypeError, ValueError):
        return {}


def _validate_document(document: RelationDocument) -> None:
    if not document.text.strip():
        raise RelationInferenceError("Relation-extraction text must not be empty.")
    for entity in document.entities:
        if (
            entity.end > len(document.text)
            or document.text[entity.start : entity.end] != entity.text
        ):
            raise RelationInferenceError("An entity does not match the supplied text and offsets.")


def _candidate_pairs(
    entities: tuple[RelationEntity, ...], maximum: int
) -> tuple[list[tuple[RelationEntity, RelationEntity]], tuple[str, ...]]:
    pairs: list[tuple[RelationEntity, RelationEntity]] = []
    skipped_overlap = False
    for source in entities:
        for target in entities:
            if source is target:
                continue
            if source.start < target.end and target.start < source.end:
                skipped_overlap = True
                continue
            pairs.append((source, target))
            if len(pairs) > maximum:
                raise RelationInferenceError(
                    "The request exceeds the configured entity-pair limit."
                )
    warnings = (
        ("Overlapping entity pairs were excluded from relation inference.",)
        if skipped_overlap
        else ()
    )
    return pairs, warnings


def _mark_entities(text: str, source: RelationEntity, target: RelationEntity) -> str:
    inserts = [
        (source.start, "[E1]"),
        (source.end, "[/E1]"),
        (target.start, "[E2]"),
        (target.end, "[/E2]"),
    ]
    marked = text
    for position, marker in sorted(inserts, reverse=True):
        marked = marked[:position] + marker + marked[position:]
    return marked

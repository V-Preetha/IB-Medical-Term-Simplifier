"""Evaluation-only providers and registry for the three NER candidates."""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any

from app.ner.contracts import (
    ENTITY_TYPES,
    BaseNERProvider,
    NERHealthStatus,
    NERProviderHealth,
    NERProviderMetadata,
    NERProviderResult,
    NormalizedEntity,
)
from app.ner.errors import (
    NERConfigurationError,
    NERInferenceError,
    NERProviderUnavailableError,
    UnsupportedNERModelError,
)
from app.ner.manifest import PENDING_APPROVAL, NERManifestEntry, NERModelManifest

logger = logging.getLogger(__name__)
ProviderFactory = Callable[[], BaseNERProvider]
_LABEL_ALIASES = {
    "disease": "Disease",
    "disorder": "Disease",
    "symptom": "Symptom",
    "sign": "Symptom",
    "medication": "Medication",
    "drug": "Medication",
    "chemical": "Medication",
    "procedure": "Procedure",
    "treatment": "Procedure",
    "anatomy": "Anatomy",
    "body_part": "Anatomy",
    "lab_test": "Laboratory Test",
    "laboratory_test": "Laboratory Test",
    "test": "Laboratory Test",
    "measurement": "Measurement",
    "value": "Measurement",
    "medical_abbreviation": "Medical Abbreviation",
    "abbreviation": "Medical Abbreviation",
}
_BIOMEDICAL_LABEL_ALIASES = {
    "disease_disorder": "Disease",
    "sign_symptom": "Symptom",
    "medication": "Medication",
    "therapeutic_procedure": "Procedure",
    "diagnostic_procedure": "Procedure",
    "biological_structure": "Anatomy",
    "lab_value": "Measurement",
    "dosage": "Measurement",
    "quantitative_concept": "Measurement",
    "mass": "Measurement",
    "volume": "Measurement",
    "weight": "Measurement",
    "height": "Measurement",
}


class NERProviderRegistry:
    """Instance-scoped registry for provider-neutral service composition."""

    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, name: str, factory: ProviderFactory) -> None:
        key = name.strip().casefold()
        if not key or key in self._factories:
            raise NERConfigurationError(f"NER provider registration is invalid: {name!r}.")
        self._factories[key] = factory
        logger.info(
            "NER provider registered",
            extra={"event": "ner_provider_registered", "provider_name": key},
        )

    def create(self, name: str) -> BaseNERProvider:
        key = name.strip().casefold()
        factory = self._factories.get(key)
        if factory is None:
            raise UnsupportedNERModelError(f"Unsupported NER provider: {name}.")
        return factory()

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


class _ManagedNERProvider(BaseNERProvider):
    def __init__(
        self,
        entry: NERManifestEntry,
        manifest_path: Path,
        *,
        device_variable: str = "NER_CONFIG__DEVICE",
        threshold_variable: str = "NER_CONFIG__CONFIDENCE_THRESHOLD",
        max_tokens_variable: str = "NER_CONFIG__MAX_TOKENS",
        stride_variable: str = "NER_CONFIG__STRIDE_TOKENS",
        label_mapping_variable: str | None = None,
    ) -> None:
        self.entry = entry
        self._manifest_path = manifest_path
        self._device_variable = device_variable
        self._model: Any = None
        self._tokenizer: Any = None
        self._pipeline: Any = None
        self._device = os.getenv(device_variable, "auto").strip().casefold()
        self._threshold = _float_environment(threshold_variable, 0.5)
        self._max_tokens = _positive_environment(max_tokens_variable, 512)
        self._stride_tokens = _nonnegative_environment(stride_variable, 64)
        if self._stride_tokens >= self._max_tokens:
            raise NERConfigurationError(
                "NER tokenizer stride must be smaller than the maximum token count."
            )
        self._label_aliases = _label_mapping(
            self.entry.key, environment_name=label_mapping_variable
        )
        self._loading_time_ms: float | None = None
        self._startup_timestamp: datetime | None = None
        self._request_count = 0
        self._lock = RLock()

    @property
    def cache_path(self) -> Path:
        override = os.getenv(_environment_name(self.entry.key, "CACHE_DIR"), "").strip()
        configured = override or self.entry.local_cache_path
        return (self._manifest_path.parent / configured).resolve()

    @property
    def repository_id(self) -> str:
        return os.getenv(
            _environment_name(self.entry.key, "REPOSITORY_ID"), self.entry.repository_id
        ).strip()

    @property
    def revision(self) -> str:
        return os.getenv(
            _environment_name(self.entry.key, "REVISION"), self.entry.pinned_revision
        ).strip()

    @property
    def configured(self) -> bool:
        return (
            self.repository_id != PENDING_APPROVAL
            and re.fullmatch(r"[0-9a-f]{40}", self.revision) is not None
            and _revision_available(self.cache_path, self.revision)
        )

    async def initialize(self) -> None:
        if self._model is not None:
            return
        if not self.configured:
            raise NERConfigurationError(
                f"NER candidate {self.entry.key} is awaiting an approved immutable "
                "checkpoint in MODEL_MANIFEST.md or its documented environment variables."
            )
        started = perf_counter()
        try:
            await asyncio.to_thread(self._load_runtime)
        except NERConfigurationError:
            raise
        except Exception as exc:
            raise NERProviderUnavailableError(
                f"NER candidate {self.entry.key} could not initialize from its local checkpoint."
            ) from exc
        self._loading_time_ms = round((perf_counter() - started) * 1000, 3)
        self._startup_timestamp = datetime.now(UTC)
        logger.info(
            "NER provider initialized",
            extra={
                "event": "ner_provider_initialized",
                "provider_name": self.entry.key,
                "model_revision": self.revision,
                "device": self._device,
                "model_loading_time_ms": self._loading_time_ms,
            },
        )

    def metadata(self) -> NERProviderMetadata:
        return NERProviderMetadata(
            provider_name=self.entry.key,
            model_name=self.entry.model_name,
            model_revision=self.revision,
            framework=self.entry.framework,
            device=self._device,
            loading_time_ms=self._loading_time_ms,
            startup_timestamp=self._startup_timestamp,
            configuration={
                "local_files_only": True,
                "cache_path": str(self.cache_path),
                "confidence_threshold": self._threshold,
                "max_tokens": self._max_tokens,
                "stride_tokens": self._stride_tokens,
                "repository_id": self.repository_id,
                "label_mapping_version": "canonical-medical-entities-v1",
                "warm": self._model is not None,
                "request_count": self._request_count,
            },
        )

    def health(self) -> NERProviderHealth:
        if self._model is not None:
            status = NERHealthStatus.READY
            detail = "NER provider is initialized from its approved local checkpoint."
        elif self.configured:
            status = NERHealthStatus.NOT_INITIALIZED
            detail = "Approved local NER provider is configured but not initialized."
        else:
            status = NERHealthStatus.NOT_CONFIGURED
            detail = "Candidate checkpoint identity or local artifact is pending approval."
        return NERProviderHealth(self.entry.key, status, detail, self.metadata())

    async def shutdown(self) -> None:
        metadata = self.metadata()
        self._pipeline = None
        self._tokenizer = None
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info(
            "NER provider shut down",
            extra={
                "event": "ner_provider_shutdown",
                "provider_name": metadata.provider_name,
                "model_revision": metadata.model_revision,
                "device": metadata.device,
            },
        )

    def _require_ready(self) -> None:
        if self._model is None:
            raise NERProviderUnavailableError(f"NER candidate {self.entry.key} is not initialized.")

    def _load_runtime(self) -> None:
        raise NotImplementedError


class LocalTokenClassificationProvider(_ManagedNERProvider):
    """Local-only Hugging Face token-classification adapter."""

    def _load_runtime(self) -> None:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        if self._device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._device not in {"cpu", "cuda"}:
            raise NERConfigurationError(f"{self._device_variable} must be auto, cpu, or cuda.")
        if self._device == "cuda" and not torch.cuda.is_available():
            raise NERProviderUnavailableError(
                "CUDA was requested for NER evaluation but is unavailable to PyTorch."
            )
        dtype = (
            (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
            if self._device == "cuda"
            else torch.float32
        )

        self._tokenizer = AutoTokenizer.from_pretrained(str(self.cache_path), local_files_only=True)
        self._model = AutoModelForTokenClassification.from_pretrained(
            str(self.cache_path), local_files_only=True, dtype=dtype
        )
        self._model.to(self._device)
        self._model.eval()

    def extract(self, text: str) -> NERProviderResult:
        self._require_ready()
        self._request_count += 1
        if not text.strip():
            raise NERInferenceError("NER input text must not be empty.")
        try:
            import torch

            with self._lock:
                encoded = self._tokenizer(
                    text,
                    return_offsets_mapping=True,
                    return_overflowing_tokens=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self._max_tokens,
                    stride=self._stride_tokens,
                    padding=True,
                )
                offset_windows = encoded.pop("offset_mapping").tolist()
                encoded.pop("overflow_to_sample_mapping", None)
            id_to_label = self._model.config.id2label
            window_bounds = [_window_bounds(offsets) for offsets in offset_windows]
            entities: list[NormalizedEntity] = []
            ignored_labels: set[str] = set()
            token_count = 0
            for index, offsets in enumerate(offset_windows):
                model_inputs = {
                    key: value[index : index + 1].to(self._device) for key, value in encoded.items()
                }
                with torch.inference_mode():
                    logits = self._model(**model_inputs).logits[0]
                probabilities = torch.softmax(logits, dim=-1)
                scores, identifiers = probabilities.max(dim=-1)
                window_entities, window_ignored = _decode_token_entities(
                    text,
                    offsets,
                    identifiers.detach().cpu().tolist(),
                    scores.detach().cpu().tolist(),
                    id_to_label,
                    self._label_aliases,
                    self._threshold,
                )
                entities.extend(
                    entity
                    for entity in window_entities
                    if _window_owns_entity(entity, index, window_bounds)
                )
                ignored_labels.update(window_ignored)
                token_count += int(model_inputs["attention_mask"].sum().item())
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise NERInferenceError(
                "Token-classification model returned invalid entities."
            ) from exc
        warnings = (
            ("Ignored unmapped labels: " + ", ".join(sorted(ignored_labels)),)
            if ignored_labels
            else ()
        )
        return NERProviderResult(_deduplicate(tuple(entities)), token_count, warnings)


class BiomedicalNERProvider(LocalTokenClassificationProvider):
    """Approved production adapter for the pinned biomedical-ner-all checkpoint."""

    def __init__(self, entry: NERManifestEntry, manifest_path: Path) -> None:
        super().__init__(
            entry,
            manifest_path,
            device_variable="NER_CONFIG__DEVICE",
            threshold_variable="NER_CONFIG__CONFIDENCE_THRESHOLD",
            max_tokens_variable="NER_CONFIG__MAX_TOKENS",
            stride_variable="NER_CONFIG__STRIDE_TOKENS",
            label_mapping_variable="NER_CONFIG__LABEL_MAPPING_JSON",
        )
        if self._stride_tokens >= self._max_tokens:
            raise NERConfigurationError(
                "NER_CONFIG__STRIDE_TOKENS must be smaller than NER_CONFIG__MAX_TOKENS."
            )
        self._label_aliases = {**_BIOMEDICAL_LABEL_ALIASES, **self._label_aliases}

    @property
    def cache_path(self) -> Path:
        configured = os.getenv("NER_CONFIG__CACHE_DIR", "").strip()
        path = configured or self.entry.local_cache_path
        return (self._manifest_path.parent / path).resolve()

    @property
    def repository_id(self) -> str:
        return os.getenv("NER_CONFIG__MODEL_NAME", self.entry.repository_id).strip()

    @property
    def revision(self) -> str:
        return os.getenv("NER_CONFIG__MODEL_REVISION", self.entry.pinned_revision).strip()

    @property
    def configured(self) -> bool:
        return (
            self.repository_id == self.entry.repository_id
            and self.revision == self.entry.pinned_revision
            and _revision_available(self.cache_path, self.revision)
        )


def create_production_registry(manifest: NERModelManifest) -> NERProviderRegistry:
    """Register only the formally approved production provider."""
    selected = manifest.production_provider
    registry = NERProviderRegistry()
    registry.register(
        selected,
        lambda: BiomedicalNERProvider(manifest.candidates[selected], manifest.path),
    )
    return registry


def _environment_name(candidate: str, suffix: str) -> str:
    return f"NER_{candidate.upper().replace('-', '_')}__{suffix}"


def _float_environment(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise NERConfigurationError(f"{name} must be numeric.") from exc
    if not 0 <= value <= 1:
        raise NERConfigurationError(f"{name} must be between zero and one.")
    return value


def _positive_environment(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise NERConfigurationError(f"{name} must be an integer.") from exc
    if value < 1:
        raise NERConfigurationError(f"{name} must be positive.")
    return value


def _nonnegative_environment(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise NERConfigurationError(f"{name} must be an integer.") from exc
    if value < 0:
        raise NERConfigurationError(f"{name} must not be negative.")
    return value


def _window_bounds(offsets: list[list[int]]) -> tuple[int, int]:
    content = [(start, end) for start, end in offsets if end > start]
    return (content[0][0], content[-1][1]) if content else (0, 0)


def _window_owns_entity(
    entity: NormalizedEntity,
    index: int,
    bounds: list[tuple[int, int]],
) -> bool:
    midpoint = (entity.start + entity.end) / 2
    left = (bounds[index][0] + bounds[index - 1][1]) / 2 if index > 0 else float("-inf")
    right = (
        (bounds[index][1] + bounds[index + 1][0]) / 2 if index + 1 < len(bounds) else float("inf")
    )
    return left <= midpoint < right


def _canonical_label(value: str, aliases: Mapping[str, str] | None = None) -> str | None:
    normalized = re.sub(r"^(B|I|E|S|U|L)-", "", value, flags=re.IGNORECASE)
    key = re.sub(r"[^a-z0-9]+", "_", normalized.casefold()).strip("_")
    return (aliases or _LABEL_ALIASES).get(key)


def _normalized_entity(
    source: str, label: str, start: int, end: int, confidence: float
) -> NormalizedEntity:
    canonical = label if label in ENTITY_TYPES else _canonical_label(label)
    if canonical is None or start < 0 or end > len(source) or end <= start:
        raise ValueError("Provider entity does not match the normalized schema.")
    return NormalizedEntity(source[start:end], canonical, start, end, confidence)


def _label_mapping(candidate: str, *, environment_name: str | None = None) -> Mapping[str, str]:
    environment_name = environment_name or _environment_name(candidate, "LABEL_MAPPING_JSON")
    raw = os.getenv(environment_name, "").strip()
    if not raw:
        return _LABEL_ALIASES
    try:
        supplied = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise NERConfigurationError(f"{environment_name} must be valid JSON.") from exc
    if not isinstance(supplied, dict) or any(
        not isinstance(key, str) or value not in ENTITY_TYPES for key, value in supplied.items()
    ):
        raise NERConfigurationError(
            f"{environment_name} must map provider labels to canonical entity types."
        )
    normalized = {
        re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_"): value
        for key, value in supplied.items()
    }
    return {**_LABEL_ALIASES, **normalized}


def _revision_available(cache_path: Path, revision: str) -> bool:
    if not cache_path.is_dir():
        return False
    direct_record = cache_path / ".cache" / "huggingface" / "trees" / f"{revision}.json"
    if cache_path.name == revision or direct_record.is_file():
        return True
    metadata_directory = cache_path / ".cache" / "huggingface" / "download"
    for metadata_path in metadata_directory.glob("*.metadata"):
        try:
            recorded_revision = metadata_path.read_text(encoding="utf-8").splitlines()[0]
        except (OSError, IndexError):
            continue
        if recorded_revision == revision:
            return True
    return False


def _decode_token_entities(
    source: str,
    offsets: list[list[int]],
    identifiers: list[int],
    scores: list[float],
    id_to_label: Mapping[int, str],
    aliases: Mapping[str, str],
    threshold: float,
) -> tuple[tuple[NormalizedEntity, ...], set[str]]:
    entities: list[NormalizedEntity] = []
    ignored: set[str] = set()
    active: dict[str, Any] | None = None

    def finish() -> None:
        nonlocal active
        if active is not None:
            entities.append(
                NormalizedEntity(
                    source[active["start"] : active["end"]],
                    active["label"],
                    active["start"],
                    active["end"],
                    sum(active["scores"]) / len(active["scores"]),
                )
            )
        active = None

    for offset, identifier, score in zip(offsets, identifiers, scores, strict=True):
        start, end = offset
        while start < end and source[start].isspace():
            start += 1
        while end > start and source[end - 1].isspace():
            end -= 1
        raw_label = str(id_to_label.get(identifier, ""))
        if start == end or raw_label.casefold() == "o" or score < threshold:
            finish()
            continue
        canonical = _canonical_label(raw_label, aliases)
        if canonical is None:
            ignored.add(raw_label)
            finish()
            continue
        prefix = raw_label.split("-", 1)[0].upper() if "-" in raw_label else "B"
        continues = (
            active is not None
            and prefix in {"I", "E", "L"}
            and active["label"] == canonical
            and start <= active["end"] + 1
        )
        if not continues:
            finish()
            active = {"label": canonical, "start": start, "end": end, "scores": [score]}
        else:
            active["end"] = end
            active["scores"].append(score)
        if prefix in {"E", "S", "U", "L"}:
            finish()
    finish()
    return _deduplicate(tuple(entities)), ignored


def _deduplicate(entities: tuple[NormalizedEntity, ...]) -> tuple[NormalizedEntity, ...]:
    unique = {(item.start, item.end, item.label): item for item in entities}
    return tuple(sorted(unique.values(), key=lambda item: (item.start, item.end, item.label)))

"""Environment-backed BioLinkBERT relation configuration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from app.relation_extraction.errors import RelationConfigurationError
from app.relation_extraction.manifest import RelationModelManifest


@dataclass(frozen=True, slots=True)
class RelationSettings:
    provider: str
    model_name: str
    model_revision: str
    cache_path: Path
    device: str
    allow_cpu_fallback: bool
    confidence_threshold: float
    batch_size: int
    max_length: int
    max_entity_pairs: int
    preprocessing_version: str
    calibration_version: str
    no_relation_labels: tuple[str, ...]

    @classmethod
    def from_manifest(cls, manifest: RelationModelManifest) -> RelationSettings:
        configured_path = os.getenv("RELATION_CONFIG__CACHE_DIR", manifest.local_cache_path).strip()
        path = Path(configured_path)
        cache_path = (
            path.resolve() if path.is_absolute() else (manifest.path.parent / path).resolve()
        )
        return cls(
            provider=os.getenv("RELATION_CONFIG__PROVIDER", manifest.provider).strip(),
            model_name=os.getenv("RELATION_CONFIG__MODEL_NAME", manifest.repository_id).strip(),
            model_revision=os.getenv(
                "RELATION_CONFIG__MODEL_REVISION", manifest.pinned_revision
            ).strip(),
            cache_path=cache_path,
            device=os.getenv("RELATION_CONFIG__DEVICE", "cpu").strip().casefold(),
            allow_cpu_fallback=_boolean("RELATION_CONFIG__ALLOW_CPU_FALLBACK", True),
            confidence_threshold=_number("RELATION_CONFIG__CONFIDENCE_THRESHOLD", 0.7),
            batch_size=_positive_int("RELATION_CONFIG__BATCH_SIZE", 8),
            max_length=_positive_int("RELATION_CONFIG__MAX_LENGTH", 512),
            max_entity_pairs=_positive_int("RELATION_CONFIG__MAX_ENTITY_PAIRS", 1_000),
            preprocessing_version=os.getenv(
                "RELATION_CONFIG__PREPROCESSING_VERSION",
                manifest.preprocessing_version,
            ).strip(),
            calibration_version=os.getenv(
                "RELATION_CONFIG__CALIBRATION_VERSION", manifest.calibration_version
            ).strip(),
            no_relation_labels=_string_tuple("RELATION_CONFIG__NO_RELATION_LABELS_JSON"),
        )

    def validate(self, manifest: RelationModelManifest) -> None:
        if self.provider != manifest.provider:
            raise RelationConfigurationError(
                "RELATION_CONFIG__PROVIDER must match the approved manifest provider."
            )
        if self.model_name != manifest.repository_id:
            raise RelationConfigurationError(
                "RELATION_CONFIG__MODEL_NAME must match the approved BioLinkBERT repository."
            )
        if self.model_revision != manifest.pinned_revision:
            raise RelationConfigurationError(
                "RELATION_CONFIG__MODEL_REVISION must match the immutable manifest revision."
            )
        if self.device not in {"cpu", "cuda", "auto"}:
            raise RelationConfigurationError("RELATION_CONFIG__DEVICE must be cpu, cuda, or auto.")
        if self.max_length < 16:
            raise RelationConfigurationError("RELATION_CONFIG__MAX_LENGTH must be at least 16.")
        if not self.cache_path.is_dir():
            raise RelationConfigurationError(
                "The approved local BioLinkBERT checkpoint is unavailable."
            )
        if not _revision_available(self.cache_path, self.model_revision):
            raise RelationConfigurationError(
                "The local BioLinkBERT artifact does not prove the approved revision."
            )


def _boolean(name: str, fallback: bool) -> bool:
    value = os.getenv(name, str(fallback)).strip().casefold()
    if value not in {"true", "false", "1", "0", "yes", "no"}:
        raise RelationConfigurationError(f"{name} must be boolean.")
    return value in {"true", "1", "yes"}


def _number(name: str, fallback: float) -> float:
    try:
        value = float(os.getenv(name, str(fallback)))
    except ValueError as exc:
        raise RelationConfigurationError(f"{name} must be numeric.") from exc
    if not 0 <= value <= 1:
        raise RelationConfigurationError(f"{name} must be between zero and one.")
    return value


def _positive_int(name: str, fallback: int) -> int:
    try:
        value = int(os.getenv(name, str(fallback)))
    except ValueError as exc:
        raise RelationConfigurationError(f"{name} must be an integer.") from exc
    if value < 1:
        raise RelationConfigurationError(f"{name} must be positive.")
    return value


def _string_tuple(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return ()
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RelationConfigurationError(f"{name} must be valid JSON.") from exc
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise RelationConfigurationError(f"{name} must be a JSON string list.")
    return tuple(value.strip() for value in values)


def _revision_available(cache_path: Path, revision: str) -> bool:
    metadata_directory = cache_path / ".cache" / "huggingface" / "download"
    for metadata_path in metadata_directory.glob("*.metadata"):
        try:
            recorded = metadata_path.read_text(encoding="utf-8").splitlines()[0]
        except (OSError, IndexError):
            continue
        if recorded == revision:
            return True
    return cache_path.name == revision

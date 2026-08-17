"""Environment-backed entity-linking configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.entity_linking.errors import EntityLinkingConfigurationError
from app.entity_linking.manifest import PENDING_APPROVAL, EntityLinkingManifest


@dataclass(frozen=True, slots=True)
class EntityLinkingSettings:
    provider: str
    provider_version: str
    language_model: str
    language_model_version: str
    language_model_path: Path | None
    terminology: str
    terminology_version: str
    knowledge_base_path: Path | None
    license_accepted: bool
    confidence_threshold: float
    max_candidates: int
    ambiguity_delta: float

    @classmethod
    def from_manifest(cls, manifest: EntityLinkingManifest) -> EntityLinkingSettings:
        root = manifest.path.parent
        return cls(
            provider=_value("ENTITY_LINKING_CONFIG__PROVIDER", manifest.provider),
            provider_version=_value(
                "ENTITY_LINKING_CONFIG__SCISPACY_VERSION", manifest.provider_version
            ),
            language_model=_value("ENTITY_LINKING_CONFIG__LANGUAGE_MODEL", manifest.language_model),
            language_model_version=_value(
                "ENTITY_LINKING_CONFIG__LANGUAGE_MODEL_VERSION",
                manifest.language_model_version,
            ),
            language_model_path=_path_value(
                "ENTITY_LINKING_CONFIG__LANGUAGE_MODEL_PATH",
                manifest.language_model_path,
                root,
            ),
            terminology=manifest.terminology,
            terminology_version=_value(
                "ENTITY_LINKING_CONFIG__UMLS_RELEASE", manifest.terminology_version
            ),
            knowledge_base_path=_path_value(
                "ENTITY_LINKING_CONFIG__UMLS_KB_PATH",
                manifest.knowledge_base_path,
                root,
            ),
            license_accepted=_boolean("ENTITY_LINKING_CONFIG__UMLS_LICENSE_ACCEPTED"),
            confidence_threshold=_number("ENTITY_LINKING_CONFIG__CONFIDENCE_THRESHOLD", 0.7),
            max_candidates=_positive_int("ENTITY_LINKING_CONFIG__MAX_CANDIDATES", 5),
            ambiguity_delta=_number("ENTITY_LINKING_CONFIG__AMBIGUITY_DELTA", 0.05),
        )

    def validate(self) -> None:
        if self.provider != "scispacy-umls":
            raise EntityLinkingConfigurationError(
                "ENTITY_LINKING_CONFIG__PROVIDER must be scispacy-umls."
            )
        pending = [
            name
            for name, value in {
                "SciSpaCy version": self.provider_version,
                "language model": self.language_model,
                "language model version": self.language_model_version,
                "UMLS release": self.terminology_version,
            }.items()
            if value == PENDING_APPROVAL
        ]
        if pending:
            raise EntityLinkingConfigurationError(
                "Entity linking awaits approved immutable configuration: "
                + ", ".join(pending)
                + "."
            )
        if not self.license_accepted:
            raise EntityLinkingConfigurationError(
                "UMLS license acceptance must be explicitly configured."
            )
        for name, path in (
            ("language model", self.language_model_path),
            ("UMLS knowledge base", self.knowledge_base_path),
        ):
            if path is None or not path.is_dir() or not any(path.iterdir()):
                raise EntityLinkingConfigurationError(
                    f"The approved local {name} directory is unavailable."
                )


def _value(name: str, fallback: str) -> str:
    return os.getenv(name, fallback).strip()


def _path_value(name: str, fallback: str, root: Path) -> Path | None:
    value = _value(name, fallback)
    if value == PENDING_APPROVAL:
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _boolean(name: str) -> bool:
    return os.getenv(name, "false").strip().casefold() in {"1", "true", "yes"}


def _number(name: str, fallback: float) -> float:
    try:
        value = float(os.getenv(name, str(fallback)))
    except ValueError as exc:
        raise EntityLinkingConfigurationError(f"{name} must be numeric.") from exc
    if not 0 <= value <= 1:
        raise EntityLinkingConfigurationError(f"{name} must be between zero and one.")
    return value


def _positive_int(name: str, fallback: int) -> int:
    try:
        value = int(os.getenv(name, str(fallback)))
    except ValueError as exc:
        raise EntityLinkingConfigurationError(f"{name} must be an integer.") from exc
    if value < 1:
        raise EntityLinkingConfigurationError(f"{name} must be positive.")
    return value

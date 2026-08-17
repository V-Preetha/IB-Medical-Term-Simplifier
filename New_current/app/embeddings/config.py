"""Environment-backed medical embedding settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.embeddings.errors import EmbeddingConfigurationError
from app.embeddings.manifest import PENDING_APPROVAL, EmbeddingModelManifest


@dataclass(frozen=True, slots=True)
class EmbeddingSettings:
    provider: str
    model_name: str
    model_revision: str
    license: str
    cache_path: Path | None
    device: str
    allow_cpu_fallback: bool
    batch_size: int
    max_length: int
    normalize: bool
    pooling_method: str

    @classmethod
    def from_manifest(cls, manifest: EmbeddingModelManifest) -> EmbeddingSettings:
        path_value = os.getenv("EMBEDDING_CONFIG__CACHE_DIR", manifest.local_cache_path).strip()
        cache_path: Path | None = None
        if path_value != PENDING_APPROVAL:
            path = Path(path_value)
            cache_path = (
                path.resolve() if path.is_absolute() else (manifest.path.parent / path).resolve()
            )
        return cls(
            provider=os.getenv("EMBEDDING_CONFIG__PROVIDER", manifest.provider).strip(),
            model_name=os.getenv("EMBEDDING_CONFIG__MODEL_NAME", manifest.repository_id).strip(),
            model_revision=os.getenv(
                "EMBEDDING_CONFIG__MODEL_REVISION", manifest.pinned_revision
            ).strip(),
            license=os.getenv("EMBEDDING_CONFIG__LICENSE", manifest.license).strip(),
            cache_path=cache_path,
            device=os.getenv("EMBEDDING_CONFIG__DEVICE", "cpu").strip().casefold(),
            allow_cpu_fallback=_boolean("EMBEDDING_CONFIG__ALLOW_CPU_FALLBACK", True),
            batch_size=_positive_int("EMBEDDING_CONFIG__BATCH_SIZE", 16),
            max_length=_positive_int("EMBEDDING_CONFIG__MAX_LENGTH", 512),
            normalize=_boolean("EMBEDDING_CONFIG__NORMALIZE", True),
            pooling_method=os.getenv(
                "EMBEDDING_CONFIG__POOLING_METHOD", manifest.pooling_method
            ).strip(),
        )

    def validate(self, manifest: EmbeddingModelManifest) -> None:
        if self.provider != "bioclinical-modernbert":
            raise EmbeddingConfigurationError(
                "EMBEDDING_CONFIG__PROVIDER must be bioclinical-modernbert."
            )
        pending = [
            name
            for name, value in {
                "repository ID": self.model_name,
                "immutable revision": self.model_revision,
                "license": self.license,
            }.items()
            if value == PENDING_APPROVAL
        ]
        if pending:
            raise EmbeddingConfigurationError(
                "Medical embeddings await approved configuration: " + ", ".join(pending) + "."
            )
        if manifest.approved and (
            self.model_name != manifest.repository_id
            or self.model_revision != manifest.pinned_revision
            or self.license != manifest.license
        ):
            raise EmbeddingConfigurationError(
                "Embedding environment identity must match the approved manifest."
            )
        if self.device not in {"cpu", "cuda", "auto"}:
            raise EmbeddingConfigurationError(
                "EMBEDDING_CONFIG__DEVICE must be cpu, cuda, or auto."
            )
        if self.pooling_method != "attention-mask-mean-v1":
            raise EmbeddingConfigurationError(
                "Only the approved attention-mask-mean-v1 pooling contract is supported."
            )
        if self.max_length < 16:
            raise EmbeddingConfigurationError("EMBEDDING_CONFIG__MAX_LENGTH must be at least 16.")
        if self.cache_path is None or not self.cache_path.is_dir():
            raise EmbeddingConfigurationError(
                "The approved local BioClinical ModernBERT checkpoint is unavailable."
            )
        if not _revision_available(self.cache_path, self.model_revision):
            raise EmbeddingConfigurationError(
                "The local embedding artifact does not prove the approved revision."
            )


def _boolean(name: str, fallback: bool) -> bool:
    value = os.getenv(name, str(fallback)).strip().casefold()
    if value not in {"true", "false", "1", "0", "yes", "no"}:
        raise EmbeddingConfigurationError(f"{name} must be boolean.")
    return value in {"true", "1", "yes"}


def _positive_int(name: str, fallback: int) -> int:
    try:
        value = int(os.getenv(name, str(fallback)))
    except ValueError as exc:
        raise EmbeddingConfigurationError(f"{name} must be an integer.") from exc
    if value < 1:
        raise EmbeddingConfigurationError(f"{name} must be positive.")
    return value


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

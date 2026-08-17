"""Local-only BioClinical ModernBERT embedding provider and registry."""

from __future__ import annotations

import asyncio
import gc
import importlib.metadata
import logging
import math
from collections.abc import Callable
from datetime import UTC, datetime
from threading import RLock
from time import perf_counter
from typing import Any

from app.embeddings.config import EmbeddingSettings
from app.embeddings.contracts import (
    BaseEmbeddingProvider,
    EmbeddingHealthStatus,
    EmbeddingInput,
    EmbeddingProviderHealth,
    EmbeddingProviderMetadata,
    EmbeddingVector,
    ProviderEmbeddingResult,
)
from app.embeddings.errors import (
    EmbeddingConfigurationError,
    EmbeddingInferenceError,
    EmbeddingProviderUnavailableError,
    UnsupportedEmbeddingProviderError,
)
from app.embeddings.manifest import EmbeddingModelManifest

logger = logging.getLogger(__name__)
ProviderFactory = Callable[[], BaseEmbeddingProvider]


class EmbeddingProviderRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, name: str, factory: ProviderFactory) -> None:
        key = name.strip().casefold()
        if not key or key in self._factories:
            raise EmbeddingConfigurationError(
                f"Embedding provider registration is invalid: {name!r}."
            )
        self._factories[key] = factory
        logger.info(
            "Embedding provider registered",
            extra={"event": "embedding_provider_registered", "provider_name": key},
        )

    def create(self, name: str) -> BaseEmbeddingProvider:
        factory = self._factories.get(name.strip().casefold())
        if factory is None:
            raise UnsupportedEmbeddingProviderError(f"Unsupported embedding provider: {name}.")
        return factory()


class BioClinicalModernBERTProvider(BaseEmbeddingProvider):
    """Mean-pooled contextual embeddings from an approved local encoder."""

    def __init__(self, settings: EmbeddingSettings, manifest: EmbeddingModelManifest) -> None:
        self._settings = settings
        self._manifest = manifest
        self._tokenizer: Any = None
        self._model: Any = None
        self._device = settings.device
        self._dimensions: int | None = None
        self._loading_time_ms: float | None = None
        self._startup_timestamp: datetime | None = None
        self._health_detail = "Provider has not been initialized."
        self._request_count = 0
        self._lock = RLock()

    async def initialize(self) -> None:
        if self._model is not None:
            return
        self._settings.validate(self._manifest)
        started = perf_counter()
        try:
            await asyncio.to_thread(self._load_runtime)
        except (EmbeddingConfigurationError, EmbeddingProviderUnavailableError):
            raise
        except Exception as exc:
            self._health_detail = "BioClinical ModernBERT initialization failed."
            raise EmbeddingProviderUnavailableError(
                "BioClinical ModernBERT could not initialize from approved local artifacts."
            ) from exc
        self._loading_time_ms = round((perf_counter() - started) * 1_000, 3)
        self._startup_timestamp = datetime.now(UTC)
        self._health_detail = "BioClinical ModernBERT embedding provider is ready."
        logger.info(
            "Embedding provider initialized",
            extra={
                "event": "embedding_provider_initialized",
                "provider_name": self._manifest.provider,
                "model_name": self._settings.model_name,
                "model_revision": self._settings.model_revision,
                "device": self._device,
                "dimensions": self._dimensions,
                "model_loading_time_ms": self._loading_time_ms,
            },
        )

    def embed_batch(self, inputs: tuple[EmbeddingInput, ...]) -> ProviderEmbeddingResult:
        if self._model is None:
            raise EmbeddingProviderUnavailableError(
                "BioClinical ModernBERT embedding provider is not initialized."
            )
        if not inputs or any(not item.text.strip() for item in inputs):
            raise EmbeddingInferenceError("Embedding input must contain non-empty text records.")
        self._request_count += 1
        vectors: list[EmbeddingVector] = []
        try:
            import torch
            import torch.nn.functional as functional

            for offset in range(0, len(inputs), self._settings.batch_size):
                batch = inputs[offset : offset + self._settings.batch_size]
                with self._lock:
                    encoded = self._tokenizer(
                        [item.text for item in batch],
                        padding=True,
                        truncation=True,
                        max_length=self._settings.max_length,
                        return_tensors="pt",
                    )
                    token_counts = encoded["attention_mask"].sum(dim=1).tolist()
                    model_inputs = {key: value.to(self._device) for key, value in encoded.items()}
                    with torch.inference_mode():
                        hidden = self._model(**model_inputs).last_hidden_state
                        pooled = _attention_mask_mean(hidden, model_inputs["attention_mask"])
                        if self._settings.normalize:
                            pooled = functional.normalize(pooled, p=2, dim=1)
                rows = pooled.detach().cpu().float().tolist()
                for item, values, token_count in zip(batch, rows, token_counts, strict=True):
                    vector = tuple(round(float(value), 8) for value in values)
                    vectors.append(
                        EmbeddingVector(
                            input_id=item.input_id,
                            values=vector,
                            token_count=int(token_count),
                            vector_norm=round(math.sqrt(sum(value * value for value in vector)), 8),
                        )
                    )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise EmbeddingInferenceError(
                "BioClinical ModernBERT returned invalid embedding output."
            ) from exc
        return ProviderEmbeddingResult(tuple(vectors))

    def metadata(self) -> EmbeddingProviderMetadata:
        try:
            provider_version = importlib.metadata.version("transformers")
        except importlib.metadata.PackageNotFoundError:
            provider_version = "unavailable"
        return EmbeddingProviderMetadata(
            provider_name=self._manifest.provider,
            provider_version=provider_version,
            model_name=self._settings.model_name,
            model_revision=self._settings.model_revision,
            framework=self._manifest.framework,
            device=self._device,
            dimensions=self._dimensions,
            pooling_method=self._settings.pooling_method,
            normalized=self._settings.normalize,
            startup_timestamp=self._startup_timestamp,
            loading_time_ms=self._loading_time_ms,
            configuration={
                "local_files_only": True,
                "batch_size": self._settings.batch_size,
                "max_length": self._settings.max_length,
                "allow_cpu_fallback": self._settings.allow_cpu_fallback,
                "cache_path": str(self._settings.cache_path or ""),
                "warm": self._model is not None,
                "request_count": self._request_count,
            },
        )

    def health(self) -> EmbeddingProviderHealth:
        if self._model is not None:
            return EmbeddingProviderHealth(
                EmbeddingHealthStatus.READY, self._health_detail, self.metadata()
            )
        try:
            self._settings.validate(self._manifest)
        except EmbeddingConfigurationError as exc:
            return EmbeddingProviderHealth(
                EmbeddingHealthStatus.NOT_CONFIGURED, exc.message, self.metadata()
            )
        return EmbeddingProviderHealth(
            EmbeddingHealthStatus.NOT_INITIALIZED,
            self._health_detail,
            self.metadata(),
        )

    async def shutdown(self) -> None:
        was_initialized = self._model is not None
        self._model = None
        self._tokenizer = None
        self._dimensions = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        if was_initialized:
            logger.info(
                "Embedding provider shut down",
                extra={
                    "event": "embedding_provider_shutdown",
                    "provider_name": self._manifest.provider,
                    "model_revision": self._settings.model_revision,
                },
            )

    def _load_runtime(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self._device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self._device == "cuda" and not torch.cuda.is_available():
            if not self._settings.allow_cpu_fallback:
                raise EmbeddingProviderUnavailableError(
                    "CUDA was requested for embeddings but is unavailable."
                )
            self._device = "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self._settings.cache_path), local_files_only=True
        )
        self._model = AutoModel.from_pretrained(
            str(self._settings.cache_path), local_files_only=True
        )
        self._model.to(self._device)
        self._model.eval()
        self._dimensions = int(self._model.config.hidden_size)


def create_production_registry(
    settings: EmbeddingSettings, manifest: EmbeddingModelManifest
) -> EmbeddingProviderRegistry:
    registry = EmbeddingProviderRegistry()
    registry.register(
        manifest.provider,
        lambda: BioClinicalModernBERTProvider(settings, manifest),
    )
    return registry


def _attention_mask_mean(hidden_states: Any, attention_mask: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

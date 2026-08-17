"""Provider-neutral medical embedding contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class EmbeddingHealthStatus(StrEnum):
    READY = "ready"
    NOT_CONFIGURED = "not_configured"
    NOT_INITIALIZED = "not_initialized"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class EmbeddingInput:
    input_id: str
    text: str


@dataclass(frozen=True, slots=True)
class EmbeddingVector:
    input_id: str
    values: tuple[float, ...]
    token_count: int
    vector_norm: float


@dataclass(frozen=True, slots=True)
class ProviderEmbeddingResult:
    embeddings: tuple[EmbeddingVector, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EmbeddingProviderMetadata:
    provider_name: str
    provider_version: str
    model_name: str
    model_revision: str
    framework: str
    device: str
    dimensions: int | None
    pooling_method: str
    normalized: bool
    startup_timestamp: datetime | None
    loading_time_ms: float | None
    configuration: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EmbeddingProviderHealth:
    status: EmbeddingHealthStatus
    detail: str
    metadata: EmbeddingProviderMetadata


class BaseEmbeddingProvider(ABC):
    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    def embed_batch(self, inputs: tuple[EmbeddingInput, ...]) -> ProviderEmbeddingResult: ...

    @abstractmethod
    def metadata(self) -> EmbeddingProviderMetadata: ...

    @abstractmethod
    def health(self) -> EmbeddingProviderHealth: ...

    @abstractmethod
    async def shutdown(self) -> None: ...

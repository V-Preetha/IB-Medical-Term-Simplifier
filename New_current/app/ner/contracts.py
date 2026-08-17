"""Provider-neutral contracts for production medical NER."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class NERHealthStatus(StrEnum):
    READY = "ready"
    NOT_INITIALIZED = "not_initialized"
    NOT_CONFIGURED = "not_configured"
    UNAVAILABLE = "unavailable"


class EntityType(StrEnum):
    DISEASE = "Disease"
    SYMPTOM = "Symptom"
    MEDICATION = "Medication"
    PROCEDURE = "Procedure"
    ANATOMY = "Anatomy"
    LABORATORY_TEST = "Laboratory Test"
    MEASUREMENT = "Measurement"
    MEDICAL_ABBREVIATION = "Medical Abbreviation"


ENTITY_TYPES = tuple(item.value for item in EntityType)


@dataclass(frozen=True, slots=True)
class NormalizedEntity:
    text: str
    label: str
    start: int
    end: int
    confidence: float

    def __post_init__(self) -> None:
        if not self.text or self.label not in ENTITY_TYPES:
            raise ValueError("Entity text and a canonical entity label are required.")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Entity offsets must define a non-empty span.")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Entity confidence must be between zero and one.")


@dataclass(frozen=True, slots=True)
class NERProviderResult:
    entities: tuple[NormalizedEntity, ...]
    token_count: int | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NERProviderMetadata:
    provider_name: str
    model_name: str
    model_revision: str
    framework: str
    device: str
    loading_time_ms: float | None
    startup_timestamp: datetime | None
    configuration: dict[str, Any]


@dataclass(frozen=True, slots=True)
class NERProviderHealth:
    provider_name: str
    status: NERHealthStatus
    detail: str
    metadata: NERProviderMetadata


class BaseNERProvider(ABC):
    """Replaceable medical NER provider."""

    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    def extract(self, text: str) -> NERProviderResult: ...

    @abstractmethod
    def metadata(self) -> NERProviderMetadata: ...

    @abstractmethod
    def health(self) -> NERProviderHealth: ...

    @abstractmethod
    async def shutdown(self) -> None: ...

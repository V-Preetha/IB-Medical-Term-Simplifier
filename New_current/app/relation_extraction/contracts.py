"""Provider-neutral relation-extraction domain contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class RelationHealthStatus(StrEnum):
    READY = "ready"
    NOT_CONFIGURED = "not_configured"
    NOT_INITIALIZED = "not_initialized"
    INCOMPATIBLE_ARTIFACT = "incompatible_artifact"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class RelationEntity:
    text: str
    label: str
    start: int
    end: int
    confidence: float
    concept_id: str | None = None
    preferred_name: str | None = None

    def __post_init__(self) -> None:
        if not self.text.strip() or not self.label.strip():
            raise ValueError("Entity text and label are required.")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Entity offsets must define a non-empty span.")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Entity confidence must be between zero and one.")


@dataclass(frozen=True, slots=True)
class RelationDocument:
    text: str
    entities: tuple[RelationEntity, ...]


@dataclass(frozen=True, slots=True)
class ClinicalRelation:
    source: RelationEntity
    target: RelationEntity
    relation_type: str
    confidence: float
    evidence_start: int
    evidence_end: int


@dataclass(frozen=True, slots=True)
class ProviderRelationResult:
    relations: tuple[ClinicalRelation, ...]
    candidate_pair_count: int
    token_count: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RelationProviderMetadata:
    provider_name: str
    provider_version: str
    model_name: str
    model_revision: str
    framework: str
    device: str
    relation_labels: tuple[str, ...]
    confidence_method: str
    calibration_version: str
    preprocessing_version: str
    startup_timestamp: datetime | None
    loading_time_ms: float | None
    configuration: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RelationProviderHealth:
    status: RelationHealthStatus
    detail: str
    metadata: RelationProviderMetadata


class BaseRelationExtractionProvider(ABC):
    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    def extract(self, document: RelationDocument) -> ProviderRelationResult: ...

    @abstractmethod
    def metadata(self) -> RelationProviderMetadata: ...

    @abstractmethod
    def health(self) -> RelationProviderHealth: ...

    @abstractmethod
    async def shutdown(self) -> None: ...

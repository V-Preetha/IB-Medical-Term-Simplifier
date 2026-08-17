"""Provider-neutral domain contracts for canonical medical concept linking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class LinkerHealthStatus(StrEnum):
    READY = "ready"
    NOT_CONFIGURED = "not_configured"
    NOT_INITIALIZED = "not_initialized"
    UNAVAILABLE = "unavailable"
    STOPPED = "stopped"


class LinkStatus(StrEnum):
    LINKED = "linked"
    AMBIGUOUS = "ambiguous"
    UNLINKED = "unlinked"


@dataclass(frozen=True, slots=True)
class SourceEntity:
    text: str
    label: str
    start: int
    end: int
    confidence: float

    def __post_init__(self) -> None:
        if not self.text.strip() or not self.label.strip():
            raise ValueError("Entity text and label are required.")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Entity offsets must define a non-empty span.")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Entity confidence must be between zero and one.")


@dataclass(frozen=True, slots=True)
class ConceptCandidate:
    concept_id: str
    preferred_name: str
    semantic_types: tuple[str, ...]
    confidence: float
    source_ontology: str

    def __post_init__(self) -> None:
        if not self.concept_id or not self.preferred_name or not self.source_ontology:
            raise ValueError("Canonical concept identity is incomplete.")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Link confidence must be between zero and one.")


@dataclass(frozen=True, slots=True)
class EntityLink:
    original_entity: SourceEntity
    status: LinkStatus
    selected_concept: ConceptCandidate | None
    candidates: tuple[ConceptCandidate, ...]
    requires_review: bool


@dataclass(frozen=True, slots=True)
class ProviderLinkResult:
    links: tuple[EntityLink, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LinkerMetadata:
    provider_name: str
    provider_version: str
    model_name: str
    model_version: str
    terminology_name: str
    terminology_version: str
    confidence_method: str
    calibration_version: str
    startup_timestamp: datetime | None
    loading_time_ms: float | None
    configuration: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LinkerHealth:
    status: LinkerHealthStatus
    detail: str
    metadata: LinkerMetadata


class BaseEntityLinkingProvider(ABC):
    """Replaceable provider contract for linking NER spans to medical concepts."""

    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    def link(self, entities: tuple[SourceEntity, ...]) -> ProviderLinkResult: ...

    @abstractmethod
    def metadata(self) -> LinkerMetadata: ...

    @abstractmethod
    def health(self) -> LinkerHealth: ...

    @abstractmethod
    async def shutdown(self) -> None: ...

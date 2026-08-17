"""Infrastructure-independent contracts for OCR pipeline providers."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any


class ProviderKind(str, Enum):
    OCR = "ocr"
    POSTPROCESSOR = "postprocessor"


class ProviderHealthStatus(str, Enum):
    NEW = "new"
    INITIALIZING = "initializing"
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class ProviderDocument:
    """Validated in-memory document passed to the multimodal OCR provider."""

    content: bytes
    file_type: str
    filename: str | None = None
    document_type: str | None = None

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError("provider document content must not be empty")
        if not self.file_type.strip():
            raise ValueError("provider document file_type must not be blank")


@dataclass(frozen=True, slots=True)
class ProviderMetadata:
    provider_name: str
    provider_version: str
    provider_kind: ProviderKind
    supported_file_types: tuple[str, ...]
    supported_document_types: tuple[str, ...]
    configuration: Mapping[str, Any]
    startup_timestamp: datetime | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "configuration", MappingProxyType(dict(self.configuration)))


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    provider_name: str
    status: ProviderHealthStatus
    checked_at: datetime
    startup_timestamp: datetime | None
    detail: str


@dataclass(frozen=True, slots=True)
class OCRProviderResult:
    text: str
    document_type: str
    confidence: float | None
    confidence_method: str | None
    processing_time_ms: float
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PostProcessingResult:
    normalized_text: str
    processing_time_ms: float
    warnings: tuple[str, ...] = ()


class BaseOCRProvider(ABC):
    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    def process(self, document: ProviderDocument) -> OCRProviderResult: ...

    @abstractmethod
    def metadata(self) -> ProviderMetadata: ...

    @abstractmethod
    def health(self) -> ProviderHealth: ...

    @abstractmethod
    async def shutdown(self) -> None: ...


class BasePostProcessor(ABC):
    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    def normalize(self, text: str, *, document_type: str) -> PostProcessingResult: ...

    @abstractmethod
    def metadata(self) -> ProviderMetadata: ...

    @abstractmethod
    def health(self) -> ProviderHealth: ...

    @abstractmethod
    async def shutdown(self) -> None: ...


Provider = BaseOCRProvider | BasePostProcessor
ProviderType = type[BaseOCRProvider] | type[BasePostProcessor]

"""Provider-neutral contracts for medical report simplification."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class MedicalTermExplanation:
    term: str
    explanation: str


@dataclass(frozen=True, slots=True)
class SimplificationLevelResult:
    level: str
    simplified_report: str
    medical_terms_explained: tuple[MedicalTermExplanation, ...]
    important_findings: tuple[str, ...]
    suggested_questions_for_doctor: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProviderSimplificationResult:
    levels: tuple[SimplificationLevelResult, ...]
    prompt_tokens: int
    output_tokens: int
    generation_time_ms: float


@dataclass(frozen=True, slots=True)
class SimplificationProviderMetadata:
    provider_name: str
    model_name: str
    model_revision: str
    prompt_version: str
    device: str
    ready: bool
    detail: str
    configuration: dict[str, Any]
    model_loading_time_ms: float | None = None


class BaseSimplificationProvider(ABC):
    @abstractmethod
    async def initialize(self, *, strict: bool = True) -> None: ...

    @abstractmethod
    def simplify(
        self,
        text: str,
        entities: dict[str, tuple[str, ...]],
        linked_concepts: tuple[dict[str, str], ...] = (),
    ) -> ProviderSimplificationResult: ...

    @abstractmethod
    def metadata(self) -> SimplificationProviderMetadata: ...

    @abstractmethod
    async def shutdown(self) -> None: ...

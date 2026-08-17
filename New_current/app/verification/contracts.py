"""Provider-neutral contracts for medical verification."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class VerificationInference:
    label: str
    probabilities: dict[str, float]
    latency_ms: float


@dataclass(frozen=True, slots=True)
class VerificationProviderMetadata:
    model_name: str
    model_revision: str
    device: str
    ready: bool
    detail: str
    configuration: dict[str, Any]


class BaseVerificationProvider(ABC):
    @abstractmethod
    async def initialize(self, *, strict: bool = True) -> None: ...

    @abstractmethod
    def infer(self, premise: str, hypothesis: str) -> VerificationInference: ...

    @abstractmethod
    def metadata(self) -> VerificationProviderMetadata: ...

    @abstractmethod
    async def shutdown(self) -> None: ...

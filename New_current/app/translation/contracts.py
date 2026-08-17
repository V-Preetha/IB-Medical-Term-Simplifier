"""Provider-neutral translation contracts."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TranslationProviderMetadata:
    provider_name: str
    model_name: str
    model_revision: str
    device: str
    ready: bool
    detail: str
    configuration: dict[str, Any]


class BaseTranslationProvider(ABC):
    @abstractmethod
    async def initialize(self, *, strict: bool = True) -> None: ...

    @abstractmethod
    def translate(self, text: str, source_language: str, target_language: str) -> str: ...

    def translate_batch(
        self, texts: tuple[str, ...], source_language: str, target_language: str
    ) -> tuple[str, ...]:
        """Translate several texts. Providers may override this with one batched call."""

        return tuple(self.translate(text, source_language, target_language) for text in texts)

    @abstractmethod
    def metadata(self) -> TranslationProviderMetadata: ...

    @abstractmethod
    async def shutdown(self) -> None: ...

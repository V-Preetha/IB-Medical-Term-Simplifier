"""Provider factories, composition, and lifecycle management."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from app.ocr.providers.config import ProviderSettings
from app.ocr.providers.contracts import (
    BaseOCRProvider,
    BasePostProcessor,
    Provider,
    ProviderHealth,
    ProviderKind,
)
from app.ocr.providers.errors import ProviderError, ProviderInitializationError
from app.ocr.providers.implementations import register_builtin_providers
from app.ocr.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)


class OCRProviderFactory:
    def __init__(self, registry: ProviderRegistry) -> None:
        self._registry = registry

    def create(self, name: str, configuration: Mapping[str, str]) -> BaseOCRProvider:
        provider_type = self._registry.resolve(ProviderKind.OCR, name)
        return cast(
            BaseOCRProvider,
            _construct_provider(provider_type, configuration, ProviderKind.OCR, name),
        )


class PostProcessorFactory:
    def __init__(self, registry: ProviderRegistry) -> None:
        self._registry = registry

    def create(self, name: str, configuration: Mapping[str, str]) -> BasePostProcessor:
        provider_type = self._registry.resolve(ProviderKind.POSTPROCESSOR, name)
        return cast(
            BasePostProcessor,
            _construct_provider(provider_type, configuration, ProviderKind.POSTPROCESSOR, name),
        )


def _construct_provider(
    provider_type: type,
    configuration: Mapping[str, str],
    kind: ProviderKind,
    name: str,
) -> Provider:
    try:
        return cast(Provider, provider_type(configuration))
    except ProviderError:
        raise
    except Exception as exc:
        logger.exception(
            "OCR provider construction failed",
            extra={
                "event": "provider_construction_failed",
                "provider_kind": kind.value,
                "provider_name": name,
            },
        )
        raise ProviderInitializationError(
            f"The configured {kind.value} provider could not be constructed."
        ) from exc


@dataclass(frozen=True, slots=True)
class ProviderContainer:
    """Application-owned providers initialized and stopped as one lifecycle unit."""

    ocr_provider: BaseOCRProvider
    postprocessor: BasePostProcessor

    @property
    def providers(self) -> tuple[Provider, ...]:
        return (self.ocr_provider, self.postprocessor)

    async def initialize(self) -> None:
        initialized: list[Provider] = []
        try:
            for provider in self.providers:
                await provider.initialize()
                initialized.append(provider)
        except ProviderError:
            for provider in reversed(initialized):
                await provider.shutdown()
            raise
        except Exception as exc:
            for provider in reversed(initialized):
                await provider.shutdown()
            logger.exception("Unexpected OCR provider lifecycle failure")
            raise ProviderInitializationError(
                "The OCR provider set could not be initialized."
            ) from exc

    async def shutdown(self) -> None:
        for provider in reversed(self.providers):
            try:
                await provider.shutdown()
            except Exception:
                logger.exception(
                    "OCR provider shutdown failed",
                    extra={
                        "event": "provider_shutdown_failed",
                        "provider_name": provider.metadata().provider_name,
                    },
                )

    def health(self) -> tuple[ProviderHealth, ...]:
        return tuple(provider.health() for provider in self.providers)


def create_provider_container(
    settings: ProviderSettings,
    *,
    registry: ProviderRegistry | None = None,
    discover: bool = True,
) -> ProviderContainer:
    """Build providers exclusively through configuration-driven factories."""

    provider_registry = registry or ProviderRegistry()
    if registry is None:
        register_builtin_providers(provider_registry)
    if discover:
        provider_registry.discover()

    ocr_provider = OCRProviderFactory(provider_registry).create(
        settings.selected(ProviderKind.OCR),
        settings.configuration_for(ProviderKind.OCR),
    )
    postprocessor = PostProcessorFactory(provider_registry).create(
        settings.selected(ProviderKind.POSTPROCESSOR),
        settings.configuration_for(ProviderKind.POSTPROCESSOR),
    )
    return ProviderContainer(
        ocr_provider=ocr_provider,
        postprocessor=postprocessor,
    )

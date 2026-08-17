"""Provider registration and entry-point discovery."""

import logging
from importlib.metadata import entry_points

from app.ocr.providers.contracts import (
    BaseOCRProvider,
    BasePostProcessor,
    ProviderKind,
    ProviderType,
)
from app.ocr.providers.errors import (
    ProviderConfigurationError,
    ProviderInitializationError,
)

logger = logging.getLogger(__name__)

_BASE_TYPES: dict[ProviderKind, type] = {
    ProviderKind.OCR: BaseOCRProvider,
    ProviderKind.POSTPROCESSOR: BasePostProcessor,
}
_ENTRY_POINT_GROUPS = {
    ProviderKind.OCR: "ib_health.ocr.providers",
    ProviderKind.POSTPROCESSOR: "ib_health.ocr.postprocessors",
}


class ProviderRegistry:
    """Instance-scoped provider catalog with optional package discovery."""

    def __init__(self) -> None:
        self._providers: dict[ProviderKind, dict[str, ProviderType]] = {
            kind: {} for kind in ProviderKind
        }

    def register(self, kind: ProviderKind, name: str, provider_type: ProviderType) -> None:
        normalized_name = name.strip().casefold()
        if not normalized_name:
            raise ProviderConfigurationError("Provider registration name must not be blank.")
        expected_type = _BASE_TYPES[kind]
        if not isinstance(provider_type, type) or not issubclass(provider_type, expected_type):
            raise ProviderConfigurationError(
                f"Provider {name!r} does not implement the {kind.value} contract."
            )
        if normalized_name in self._providers[kind]:
            raise ProviderConfigurationError(
                f"Provider {normalized_name!r} is already registered for {kind.value}."
            )
        self._providers[kind][normalized_name] = provider_type
        logger.info(
            "OCR provider registered",
            extra={
                "event": "provider_registered",
                "provider_kind": kind.value,
                "provider_name": normalized_name,
            },
        )

    def resolve(self, kind: ProviderKind, name: str) -> ProviderType:
        normalized_name = name.strip().casefold()
        try:
            return self._providers[kind][normalized_name]
        except KeyError as exc:
            available = ", ".join(sorted(self._providers[kind])) or "none"
            raise ProviderConfigurationError(
                f"Unsupported {kind.value} provider {name!r}; registered providers: {available}."
            ) from exc

    def registered_names(self, kind: ProviderKind) -> tuple[str, ...]:
        return tuple(sorted(self._providers[kind]))

    def discover(self) -> None:
        """Register installed providers declared through Python package entry points."""

        discovered = entry_points()
        for kind, group in _ENTRY_POINT_GROUPS.items():
            candidates = (
                discovered.select(group=group)
                if hasattr(discovered, "select")
                else discovered.get(group, ())
            )
            for candidate in candidates:
                try:
                    provider_type = candidate.load()
                    self.register(kind, candidate.name, provider_type)
                except ProviderConfigurationError:
                    raise
                except Exception as exc:
                    logger.exception(
                        "OCR provider discovery failed",
                        extra={
                            "event": "provider_discovery_failed",
                            "provider_kind": kind.value,
                            "provider_name": candidate.name,
                        },
                    )
                    raise ProviderInitializationError(
                        f"Could not load registered provider {candidate.name!r}."
                    ) from exc

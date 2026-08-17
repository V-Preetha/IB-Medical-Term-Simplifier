"""Environment-backed provider selection and configuration."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from app.ocr.providers.contracts import ProviderKind
from app.ocr.providers.errors import ProviderConfigurationError
from app.ocr.providers.manifest import (
    load_model_manifest,
    merge_manifest_model_configuration,
)

_SELECTION_VARIABLES = {
    ProviderKind.OCR: "OCR_PROVIDER",
    ProviderKind.POSTPROCESSOR: "POSTPROCESSOR_PROVIDER",
}
_CONFIG_PREFIXES = {
    ProviderKind.OCR: "OCR_CONFIG__",
    ProviderKind.POSTPROCESSOR: "POSTPROCESSOR_CONFIG__",
}


@dataclass(frozen=True, slots=True)
class ProviderSettings:
    """Provider names and opaque provider configuration loaded from the environment."""

    selections: Mapping[ProviderKind, str]
    configurations: Mapping[ProviderKind, Mapping[str, str]]

    def __post_init__(self) -> None:
        selections = {kind: value.strip().casefold() for kind, value in self.selections.items()}
        missing = [kind.value for kind in ProviderKind if not selections.get(kind)]
        if missing:
            raise ProviderConfigurationError(
                "Provider selection is required for: " + ", ".join(missing)
            )
        configurations = {
            kind: MappingProxyType(dict(self.configurations.get(kind, {}))) for kind in ProviderKind
        }
        object.__setattr__(self, "selections", MappingProxyType(selections))
        object.__setattr__(self, "configurations", MappingProxyType(configurations))

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
        *,
        manifest_path: Path | None = None,
    ) -> "ProviderSettings":
        source = os.environ if environment is None else environment
        manifest = load_model_manifest(manifest_path)
        selections: dict[ProviderKind, str] = {}
        configurations: dict[ProviderKind, dict[str, str]] = {}
        for kind in ProviderKind:
            variable = _SELECTION_VARIABLES[kind]
            value = source.get(variable, "").strip()
            if not value:
                raise ProviderConfigurationError(
                    f"Required environment variable {variable} is not configured."
                )
            selections[kind] = value
            prefix = _CONFIG_PREFIXES[kind]
            environment_configuration = {
                key.removeprefix(prefix).casefold(): config_value.strip()
                for key, config_value in source.items()
                if key.startswith(prefix)
            }
            configurations[kind] = merge_manifest_model_configuration(
                kind,
                environment_configuration,
                manifest=manifest,
            )
        return cls(selections=selections, configurations=configurations)

    def selected(self, kind: ProviderKind) -> str:
        return self.selections[kind]

    def configuration_for(self, kind: ProviderKind) -> Mapping[str, str]:
        return self.configurations[kind]

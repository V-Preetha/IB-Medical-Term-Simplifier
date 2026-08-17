"""Deterministic reader for the repository-owned OCR model manifest."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from app.ocr.providers.contracts import ProviderKind
from app.ocr.providers.errors import ProviderConfigurationError

PENDING_APPROVAL = "PENDING_APPROVAL"
DEFAULT_MODEL_MANIFEST_PATH = Path(__file__).resolve().parents[4] / "MODEL_MANIFEST.md"
_BEGIN_MARKER = "<!-- MODEL_MANIFEST_DATA_BEGIN -->"
_END_MARKER = "<!-- MODEL_MANIFEST_DATA_END -->"
_MODEL_KEYS = {
    ProviderKind.OCR: "qwen3-vl",
}
_REQUIRED_FIELDS = {
    "model_name",
    "purpose",
    "provider",
    "repository_id",
    "pinned_revision",
    "license",
    "local_cache_path",
    "expected_sha256",
    "device",
    "configuration_variables",
}


@dataclass(frozen=True, slots=True)
class ModelManifestEntry:
    """One approved-or-pending model inventory entry."""

    key: str
    model_name: str
    purpose: str
    provider: str
    repository_id: str
    pinned_revision: str
    license: str
    local_cache_path: str
    expected_sha256: str
    device: str
    configuration_variables: tuple[str, ...]

    @property
    def approval_pending(self) -> bool:
        return any(
            value == PENDING_APPROVAL
            for value in (
                self.repository_id,
                self.pinned_revision,
                self.license,
            )
        )


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Validated model inventory loaded from ``MODEL_MANIFEST.md``."""

    path: Path
    schema_version: str
    models: Mapping[str, ModelManifestEntry]

    def entry_for(self, kind: ProviderKind) -> ModelManifestEntry:
        key = _MODEL_KEYS.get(kind)
        if key is None:
            raise ProviderConfigurationError(
                f"The model manifest does not define a model for {kind.value}."
            )
        return self.models[key]


def load_model_manifest(path: Path | None = None) -> ModelManifest:
    """Load and validate the machine-readable contract embedded in the manifest."""

    manifest_path = (path or DEFAULT_MODEL_MANIFEST_PATH).resolve()
    try:
        contents = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProviderConfigurationError(
            f"Model manifest is unavailable at {manifest_path}."
        ) from exc
    if _BEGIN_MARKER not in contents or _END_MARKER not in contents:
        raise ProviderConfigurationError("Model manifest data markers are missing.")
    embedded = contents.split(_BEGIN_MARKER, 1)[1].split(_END_MARKER, 1)[0]
    match = re.search(r"```json\s*(\{.*\})\s*```", embedded, flags=re.DOTALL)
    if match is None:
        raise ProviderConfigurationError("Model manifest JSON block is missing.")
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ProviderConfigurationError("Model manifest JSON is invalid.") from exc
    return _validate_manifest_payload(manifest_path, payload)


def merge_manifest_model_configuration(
    kind: ProviderKind,
    configuration: Mapping[str, str],
    *,
    manifest: ModelManifest,
) -> dict[str, str]:
    """Fill missing model identity settings from the manifest and fail on pending data."""

    merged = dict(configuration)
    if kind not in _MODEL_KEYS:
        return merged
    entry = manifest.entry_for(kind)
    sources = {
        "model_name": (entry.repository_id, "repository_id"),
        "model_revision": (entry.pinned_revision, "pinned_revision"),
        "hf_cache_dir": (
            str((manifest.path.parent / entry.local_cache_path).resolve()),
            "local_cache_path",
        ),
        "device": (entry.device, "device"),
    }
    for configuration_key, (manifest_value, manifest_field) in sources.items():
        if merged.get(configuration_key, "").strip():
            continue
        if manifest_value == PENDING_APPROVAL:
            raise ProviderConfigurationError(
                f"MODEL_MANIFEST.md has PENDING_APPROVAL for {entry.key}.{manifest_field}; "
                f"configure an approved value through {entry.configuration_variables}."
            )
        merged[configuration_key] = manifest_value
    return merged


def _validate_manifest_payload(path: Path, payload: Any) -> ModelManifest:
    if not isinstance(payload, dict) or payload.get("schema_version") != "1":
        raise ProviderConfigurationError("Model manifest schema_version must be '1'.")
    raw_models = payload.get("models")
    if not isinstance(raw_models, dict) or set(raw_models) != set(_MODEL_KEYS.values()):
        raise ProviderConfigurationError(
            "Model manifest must define only the qwen3-vl model entry."
        )
    models = {key: _validate_entry(key, value) for key, value in raw_models.items()}
    return ModelManifest(
        path=path,
        schema_version="1",
        models=MappingProxyType(models),
    )


def _validate_entry(key: str, raw: Any) -> ModelManifestEntry:
    if not isinstance(raw, dict) or set(raw) != _REQUIRED_FIELDS:
        raise ProviderConfigurationError(
            f"Model manifest entry {key} does not contain the required fields."
        )
    values = {field: raw[field] for field in _REQUIRED_FIELDS}
    text_fields = _REQUIRED_FIELDS - {"configuration_variables"}
    if any(
        not isinstance(values[field], str) or not values[field].strip() for field in text_fields
    ):
        raise ProviderConfigurationError(f"Model manifest entry {key} contains a blank field.")
    variables = values["configuration_variables"]
    if (
        not isinstance(variables, list)
        or not variables
        or any(not isinstance(item, str) or not item.strip() for item in variables)
    ):
        raise ProviderConfigurationError(
            f"Model manifest entry {key} has invalid configuration_variables."
        )
    revision = values["pinned_revision"]
    if revision.casefold() in {"latest", "main", "master", "head"}:
        raise ProviderConfigurationError(f"Model manifest entry {key} uses a moving revision.")
    if revision != PENDING_APPROVAL and re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ProviderConfigurationError(
            f"Model manifest entry {key} pinned_revision must be a 40-character commit SHA."
        )
    checksum = values["expected_sha256"]
    if checksum != PENDING_APPROVAL and re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
        raise ProviderConfigurationError(
            f"Model manifest entry {key} expected_sha256 must be a lowercase SHA-256 digest."
        )
    return ModelManifestEntry(
        key=key,
        model_name=values["model_name"],
        purpose=values["purpose"],
        provider=values["provider"],
        repository_id=values["repository_id"],
        pinned_revision=revision,
        license=values["license"],
        local_cache_path=values["local_cache_path"],
        expected_sha256=checksum,
        device=values["device"],
        configuration_variables=tuple(variables),
    )

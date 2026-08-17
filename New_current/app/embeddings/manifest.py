"""Fail-closed BioClinical ModernBERT manifest reader."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.embeddings.errors import EmbeddingConfigurationError

PENDING_APPROVAL = "PENDING_APPROVAL"
DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parents[3] / "MODEL_MANIFEST.md"
_BEGIN = "<!-- EMBEDDING_MANIFEST_DATA_BEGIN -->"
_END = "<!-- EMBEDDING_MANIFEST_DATA_END -->"
_FIELDS = {
    "model_name",
    "purpose",
    "provider",
    "framework",
    "repository_id",
    "pinned_revision",
    "license",
    "local_cache_path",
    "expected_sha256",
    "pooling_method",
    "configuration_variables",
}


@dataclass(frozen=True, slots=True)
class EmbeddingModelManifest:
    path: Path
    model_name: str
    purpose: str
    provider: str
    framework: str
    repository_id: str
    pinned_revision: str
    license: str
    local_cache_path: str
    expected_sha256: str
    pooling_method: str
    configuration_variables: tuple[str, ...]

    @property
    def approved(self) -> bool:
        return all(
            value != PENDING_APPROVAL
            for value in (
                self.repository_id,
                self.pinned_revision,
                self.license,
                self.local_cache_path,
            )
        )


def load_embedding_model_manifest(path: Path | None = None) -> EmbeddingModelManifest:
    manifest_path = (path or DEFAULT_MANIFEST_PATH).resolve()
    try:
        contents = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EmbeddingConfigurationError(
            f"Model manifest is unavailable at {manifest_path}."
        ) from exc
    if _BEGIN not in contents or _END not in contents:
        raise EmbeddingConfigurationError("Embedding manifest data markers are missing.")
    embedded = contents.split(_BEGIN, 1)[1].split(_END, 1)[0]
    match = re.search(r"```json\s*(\{.*\})\s*```", embedded, flags=re.DOTALL)
    if match is None:
        raise EmbeddingConfigurationError("Embedding manifest JSON block is missing.")
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise EmbeddingConfigurationError("Embedding manifest JSON is invalid.") from exc
    return _validate_payload(manifest_path, payload)


def _validate_payload(path: Path, payload: Any) -> EmbeddingModelManifest:
    if not isinstance(payload, dict) or payload.get("schema_version") != "1":
        raise EmbeddingConfigurationError("Embedding manifest schema_version must be '1'.")
    production = payload.get("production")
    if not isinstance(production, dict) or set(production) != _FIELDS:
        raise EmbeddingConfigurationError("Embedding production manifest fields are invalid.")
    if any(
        not isinstance(production[field], str) or not production[field].strip()
        for field in _FIELDS - {"configuration_variables"}
    ):
        raise EmbeddingConfigurationError("Embedding manifest contains a blank field.")
    if production["provider"] != "bioclinical-modernbert":
        raise EmbeddingConfigurationError(
            "The approved embedding provider must be bioclinical-modernbert."
        )
    revision = production["pinned_revision"]
    if revision != PENDING_APPROVAL and re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise EmbeddingConfigurationError("The embedding revision must be an immutable commit SHA.")
    checksum = production["expected_sha256"]
    if checksum != PENDING_APPROVAL and re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
        raise EmbeddingConfigurationError("The embedding checksum is invalid.")
    variables = production["configuration_variables"]
    if (
        not isinstance(variables, list)
        or not variables
        or any(not isinstance(item, str) or not item.strip() for item in variables)
    ):
        raise EmbeddingConfigurationError("Embedding configuration_variables are invalid.")
    return EmbeddingModelManifest(
        path=path,
        model_name=production["model_name"],
        purpose=production["purpose"],
        provider=production["provider"],
        framework=production["framework"],
        repository_id=production["repository_id"],
        pinned_revision=revision,
        license=production["license"],
        local_cache_path=production["local_cache_path"],
        expected_sha256=checksum,
        pooling_method=production["pooling_method"],
        configuration_variables=tuple(variables),
    )

"""Fail-closed reader for production and archived NER models."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from app.ner.errors import NERConfigurationError

PENDING_APPROVAL = "PENDING_APPROVAL"
DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parents[3] / "MODEL_MANIFEST.md"
_BEGIN = "<!-- NER_MODEL_MANIFEST_DATA_BEGIN -->"
_END = "<!-- NER_MODEL_MANIFEST_DATA_END -->"
_CANDIDATES = ("openmed-gliner", "biomedical-ner-all", "modernbert-biomedical-ner")
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
    "configuration_variables",
}


@dataclass(frozen=True, slots=True)
class NERManifestEntry:
    key: str
    model_name: str
    purpose: str
    provider: str
    framework: str
    repository_id: str
    pinned_revision: str
    license: str
    local_cache_path: str
    expected_sha256: str
    configuration_variables: tuple[str, ...]

    @property
    def approved(self) -> bool:
        return all(
            value != PENDING_APPROVAL
            for value in (self.repository_id, self.pinned_revision, self.license)
        )


@dataclass(frozen=True, slots=True)
class NERModelManifest:
    path: Path
    schema_version: str
    production_provider: str
    candidates: Mapping[str, NERManifestEntry]


def load_ner_model_manifest(path: Path | None = None) -> NERModelManifest:
    manifest_path = (path or DEFAULT_MANIFEST_PATH).resolve()
    try:
        contents = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise NERConfigurationError(f"Model manifest is unavailable at {manifest_path}.") from exc
    if _BEGIN not in contents or _END not in contents:
        raise NERConfigurationError("NER model manifest data markers are missing.")
    embedded = contents.split(_BEGIN, 1)[1].split(_END, 1)[0]
    match = re.search(r"```json\s*(\{.*\})\s*```", embedded, flags=re.DOTALL)
    if match is None:
        raise NERConfigurationError("NER model manifest JSON block is missing.")
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise NERConfigurationError("NER model manifest JSON is invalid.") from exc
    return _validate_payload(manifest_path, payload)


def _validate_payload(path: Path, payload: Any) -> NERModelManifest:
    if not isinstance(payload, dict) or payload.get("schema_version") != "1":
        raise NERConfigurationError("NER model manifest schema_version must be '1'.")
    production_provider = payload.get("production_provider")
    if production_provider != "biomedical-ner-all":
        raise NERConfigurationError("NER manifest production_provider must be biomedical-ner-all.")
    candidates = payload.get("candidates")
    if not isinstance(candidates, dict) or set(candidates) != set(_CANDIDATES):
        raise NERConfigurationError("NER manifest must define exactly the three candidates.")
    entries = {key: _validate_entry(key, raw) for key, raw in candidates.items()}
    return NERModelManifest(path, "1", production_provider, MappingProxyType(entries))


def _validate_entry(key: str, raw: Any) -> NERManifestEntry:
    if not isinstance(raw, dict) or set(raw) != _FIELDS:
        raise NERConfigurationError(f"NER manifest entry {key} has invalid fields.")
    text_fields = _FIELDS - {"configuration_variables"}
    if any(not isinstance(raw[field], str) or not raw[field].strip() for field in text_fields):
        raise NERConfigurationError(f"NER manifest entry {key} contains a blank field.")
    variables = raw["configuration_variables"]
    if (
        not isinstance(variables, list)
        or not variables
        or any(not isinstance(item, str) or not item for item in variables)
    ):
        raise NERConfigurationError(
            f"NER manifest entry {key} has invalid configuration variables."
        )
    revision = raw["pinned_revision"]
    if revision.casefold() in {"latest", "main", "master", "head"}:
        raise NERConfigurationError(f"NER manifest entry {key} uses a moving revision.")
    if revision != PENDING_APPROVAL and re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise NERConfigurationError(
            f"NER manifest entry {key} revision must be an immutable commit SHA."
        )
    checksum = raw["expected_sha256"]
    if checksum != PENDING_APPROVAL and re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
        raise NERConfigurationError(f"NER manifest entry {key} checksum is invalid.")
    return NERManifestEntry(
        key=key,
        model_name=raw["model_name"],
        purpose=raw["purpose"],
        provider=raw["provider"],
        framework=raw["framework"],
        repository_id=raw["repository_id"],
        pinned_revision=revision,
        license=raw["license"],
        local_cache_path=raw["local_cache_path"],
        expected_sha256=checksum,
        configuration_variables=tuple(variables),
    )

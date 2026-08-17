"""Strict reader for the Phase 7 BioLinkBERT inventory."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.relation_extraction.errors import RelationConfigurationError

PENDING_APPROVAL = "PENDING_APPROVAL"
DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parents[3] / "MODEL_MANIFEST.md"
_BEGIN = "<!-- RELATION_EXTRACTION_MANIFEST_DATA_BEGIN -->"
_END = "<!-- RELATION_EXTRACTION_MANIFEST_DATA_END -->"
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
    "preprocessing_version",
    "calibration_version",
    "configuration_variables",
}


@dataclass(frozen=True, slots=True)
class RelationModelManifest:
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
    preprocessing_version: str
    calibration_version: str
    configuration_variables: tuple[str, ...]


def load_relation_model_manifest(path: Path | None = None) -> RelationModelManifest:
    manifest_path = (path or DEFAULT_MANIFEST_PATH).resolve()
    try:
        contents = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RelationConfigurationError(
            f"Model manifest is unavailable at {manifest_path}."
        ) from exc
    if _BEGIN not in contents or _END not in contents:
        raise RelationConfigurationError("Relation-extraction manifest data markers are missing.")
    embedded = contents.split(_BEGIN, 1)[1].split(_END, 1)[0]
    match = re.search(r"```json\s*(\{.*\})\s*```", embedded, flags=re.DOTALL)
    if match is None:
        raise RelationConfigurationError("Relation-extraction manifest JSON block is missing.")
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise RelationConfigurationError("Relation-extraction manifest JSON is invalid.") from exc
    return _validate_payload(manifest_path, payload)


def _validate_payload(path: Path, payload: Any) -> RelationModelManifest:
    if not isinstance(payload, dict) or payload.get("schema_version") != "1":
        raise RelationConfigurationError("Relation-extraction manifest schema_version must be '1'.")
    production = payload.get("production")
    if not isinstance(production, dict) or set(production) != _FIELDS:
        raise RelationConfigurationError(
            "Relation-extraction production manifest fields are invalid."
        )
    if any(
        not isinstance(production[field], str) or not production[field].strip()
        for field in _FIELDS - {"configuration_variables"}
    ):
        raise RelationConfigurationError("Relation-extraction manifest contains a blank field.")
    if production["provider"] != "biolinkbert":
        raise RelationConfigurationError("The approved provider must be biolinkbert.")
    if production["repository_id"] != "michiyasunaga/BioLinkBERT-base":
        raise RelationConfigurationError(
            "The approved repository must be michiyasunaga/BioLinkBERT-base."
        )
    if production["license"] != "Apache-2.0":
        raise RelationConfigurationError("The approved license must be Apache-2.0.")
    revision = production["pinned_revision"]
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RelationConfigurationError(
            "The BioLinkBERT revision must be an immutable commit SHA."
        )
    checksum = production["expected_sha256"]
    if checksum != PENDING_APPROVAL and re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
        raise RelationConfigurationError("The BioLinkBERT checksum is invalid.")
    variables = production["configuration_variables"]
    if (
        not isinstance(variables, list)
        or not variables
        or any(not isinstance(item, str) or not item.strip() for item in variables)
    ):
        raise RelationConfigurationError("Relation-extraction configuration_variables are invalid.")
    return RelationModelManifest(
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
        preprocessing_version=production["preprocessing_version"],
        calibration_version=production["calibration_version"],
        configuration_variables=tuple(variables),
    )

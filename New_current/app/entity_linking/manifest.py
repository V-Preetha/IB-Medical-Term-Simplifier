"""Fail-closed reader for the approved entity-linking inventory."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.entity_linking.errors import EntityLinkingConfigurationError

PENDING_APPROVAL = "PENDING_APPROVAL"
DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parents[3] / "MODEL_MANIFEST.md"
_BEGIN = "<!-- ENTITY_LINKING_MANIFEST_DATA_BEGIN -->"
_END = "<!-- ENTITY_LINKING_MANIFEST_DATA_END -->"
_FIELDS = {
    "provider",
    "provider_version",
    "language_model",
    "language_model_version",
    "language_model_path",
    "terminology",
    "terminology_version",
    "knowledge_base_path",
    "license",
    "configuration_variables",
}


@dataclass(frozen=True, slots=True)
class EntityLinkingManifest:
    path: Path
    provider: str
    provider_version: str
    language_model: str
    language_model_version: str
    language_model_path: str
    terminology: str
    terminology_version: str
    knowledge_base_path: str
    license: str
    configuration_variables: tuple[str, ...]

    @property
    def approved(self) -> bool:
        return all(
            value != PENDING_APPROVAL
            for value in (
                self.provider_version,
                self.language_model,
                self.language_model_version,
                self.language_model_path,
                self.terminology_version,
                self.knowledge_base_path,
                self.license,
            )
        )


def load_entity_linking_manifest(path: Path | None = None) -> EntityLinkingManifest:
    manifest_path = (path or DEFAULT_MANIFEST_PATH).resolve()
    try:
        contents = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EntityLinkingConfigurationError(
            f"Model manifest is unavailable at {manifest_path}."
        ) from exc
    if _BEGIN not in contents or _END not in contents:
        raise EntityLinkingConfigurationError(
            "Entity-linking model manifest data markers are missing."
        )
    embedded = contents.split(_BEGIN, 1)[1].split(_END, 1)[0]
    match = re.search(r"```json\s*(\{.*\})\s*```", embedded, flags=re.DOTALL)
    if match is None:
        raise EntityLinkingConfigurationError(
            "Entity-linking model manifest JSON block is missing."
        )
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise EntityLinkingConfigurationError(
            "Entity-linking model manifest JSON is invalid."
        ) from exc
    return _validate_payload(manifest_path, payload)


def _validate_payload(path: Path, payload: Any) -> EntityLinkingManifest:
    if not isinstance(payload, dict) or payload.get("schema_version") != "1":
        raise EntityLinkingConfigurationError("Entity-linking manifest schema_version must be '1'.")
    production = payload.get("production")
    if not isinstance(production, dict) or set(production) != _FIELDS:
        raise EntityLinkingConfigurationError(
            "Entity-linking manifest production entry has invalid fields."
        )
    text_fields = _FIELDS - {"configuration_variables"}
    if any(
        not isinstance(production[field], str) or not production[field].strip()
        for field in text_fields
    ):
        raise EntityLinkingConfigurationError("Entity-linking manifest contains a blank field.")
    variables = production["configuration_variables"]
    if (
        not isinstance(variables, list)
        or not variables
        or any(not isinstance(item, str) or not item.strip() for item in variables)
    ):
        raise EntityLinkingConfigurationError(
            "Entity-linking configuration_variables must be a non-empty string list."
        )
    if production["provider"] != "scispacy-umls":
        raise EntityLinkingConfigurationError(
            "The approved entity-linking provider must be scispacy-umls."
        )
    if production["terminology"] != "UMLS":
        raise EntityLinkingConfigurationError(
            "The approved entity-linking terminology must be UMLS."
        )
    return EntityLinkingManifest(
        path=path,
        provider=production["provider"],
        provider_version=production["provider_version"],
        language_model=production["language_model"],
        language_model_version=production["language_model_version"],
        language_model_path=production["language_model_path"],
        terminology=production["terminology"],
        terminology_version=production["terminology_version"],
        knowledge_base_path=production["knowledge_base_path"],
        license=production["license"],
        configuration_variables=tuple(variables),
    )

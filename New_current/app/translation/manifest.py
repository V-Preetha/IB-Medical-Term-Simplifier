"""Fail-closed IndicTrans2 inventory loaded from ``MODEL_MANIFEST.md``."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from app.translation.errors import TranslationUnavailableError

PENDING_APPROVAL = "PENDING_APPROVAL"
DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parents[3] / "MODEL_MANIFEST.md"
_BEGIN = "<!-- TRANSLATION_MODEL_MANIFEST_DATA_BEGIN -->"
_END = "<!-- TRANSLATION_MODEL_MANIFEST_DATA_END -->"
_FIELDS = {
    "model_name",
    "provider",
    "repository_id",
    "pinned_revision",
    "license",
    "local_cache_path",
    "expected_sha256",
    "configuration_variables",
}


@dataclass(frozen=True, slots=True)
class TranslationModelManifest:
    model_name: str
    provider: str
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
            for value in (
                self.repository_id,
                self.pinned_revision,
                self.license,
                self.local_cache_path,
            )
        )

    @property
    def provisioning_message(self) -> str:
        missing = [
            name
            for name, value in (
                ("pinned_revision", self.pinned_revision),
                ("license", self.license),
                ("local_cache_path", self.local_cache_path),
            )
            if value == PENDING_APPROVAL
        ]
        detail = ", ".join(missing) if missing else "a valid local checkpoint"
        return f"IndicTrans2 provisioning is pending: MODEL_MANIFEST.md requires {detail}."


def load_translation_model_manifest(path: Path | None = None) -> TranslationModelManifest:
    manifest_path = (path or DEFAULT_MANIFEST_PATH).resolve()
    try:
        contents = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TranslationUnavailableError("Translation model manifest is unavailable.") from exc
    if _BEGIN not in contents or _END not in contents:
        raise TranslationUnavailableError("Translation model manifest data markers are missing.")
    embedded = contents.split(_BEGIN, 1)[1].split(_END, 1)[0]
    match = re.search(r"```json\s*(\{.*\})\s*```", embedded, flags=re.DOTALL)
    if match is None:
        raise TranslationUnavailableError("Translation model manifest JSON is missing.")
    try:
        raw = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise TranslationUnavailableError("Translation model manifest JSON is invalid.") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != "1":
        raise TranslationUnavailableError("Translation model manifest schema_version must be '1'.")
    entry = raw.get("production_model")
    if not isinstance(entry, dict) or set(entry) != _FIELDS:
        raise TranslationUnavailableError("Translation model manifest inventory is invalid.")
    values = {field: entry.get(field) for field in _FIELDS}
    if any(
        not isinstance(value, str) or not value.strip()
        for field, value in values.items()
        if field != "configuration_variables"
    ):
        raise TranslationUnavailableError("Translation model manifest contains a blank field.")
    variables = values["configuration_variables"]
    if (
        not isinstance(variables, list)
        or not variables
        or any(not isinstance(value, str) or not value for value in variables)
    ):
        raise TranslationUnavailableError("Translation configuration_variables are invalid.")
    if values["provider"] != "indictrans2":
        raise TranslationUnavailableError("Translation provider must be indictrans2.")
    if values["repository_id"] != "ai4bharat/indictrans2-en-indic-dist-200M":
        raise TranslationUnavailableError(
            "Translation repository is not the approved IndicTrans2 model."
        )
    revision = values["pinned_revision"]
    if revision.casefold() in {"latest", "main", "master", "head"} or (
        revision != PENDING_APPROVAL and re.fullmatch(r"[0-9a-f]{40}", revision) is None
    ):
        raise TranslationUnavailableError("Translation revision must be an immutable commit SHA.")
    checksum = values["expected_sha256"]
    if checksum != PENDING_APPROVAL and re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
        raise TranslationUnavailableError("Translation checkpoint checksum is invalid.")
    return TranslationModelManifest(
        model_name=values["model_name"],
        provider=values["provider"],
        repository_id=values["repository_id"],
        pinned_revision=revision,
        license=values["license"],
        local_cache_path=values["local_cache_path"],
        expected_sha256=checksum,
        configuration_variables=tuple(variables),
    )

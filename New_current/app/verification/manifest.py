"""Strict local inventory for the technically validated MedNLI candidate."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from app.verification.errors import VerificationUnavailableError

DEFAULT_MANIFEST = Path(__file__).resolve().parents[3] / "MODEL_MANIFEST.md"
BEGIN = "<!-- VERIFICATION_MODEL_MANIFEST_DATA_BEGIN -->"
END = "<!-- VERIFICATION_MODEL_MANIFEST_DATA_END -->"
REPOSITORY = "pritamdeka/PubMedBERT-MNLI-MedNLI"
REVISION = "f1b6ce2e0d49f295b4cbcdc56c01b5fab6d068ab"
LABELS = {0: "contradiction", 1: "entailment", 2: "neutral"}


@dataclass(frozen=True, slots=True)
class VerificationManifest:
    repository_id: str
    pinned_revision: str
    license: str
    local_cache_path: str


def load_manifest(path: Path | None = None) -> VerificationManifest:
    try:
        text = (path or DEFAULT_MANIFEST).read_text(encoding="utf-8")
        payload = text.split(BEGIN, 1)[1].split(END, 1)[0]
        raw = json.loads(re.search(r"json\s*(\{.*\})", payload, re.DOTALL).group(1))
    except (OSError, IndexError, AttributeError, json.JSONDecodeError) as exc:
        raise VerificationUnavailableError("Verification model manifest is invalid.") from exc
    required = {"repository_id", "pinned_revision", "license", "local_cache_path"}
    if not isinstance(raw, dict) or set(raw) != required:
        raise VerificationUnavailableError("Verification model manifest fields are invalid.")
    if raw["repository_id"] != REPOSITORY or raw["pinned_revision"] != REVISION:
        raise VerificationUnavailableError("Verification model identity does not match approval.")
    if not isinstance(raw["license"], str) or not isinstance(raw["local_cache_path"], str):
        raise VerificationUnavailableError("Verification model manifest has invalid values.")
    return VerificationManifest(**raw)

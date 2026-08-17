"""Composition root for the approved production NER service."""

import os

from app.ner.manifest import load_ner_model_manifest
from app.ner.providers import create_production_registry
from app.ner.service import MedicalNERService


def create_ner_service() -> MedicalNERService:
    """Create the single-provider production NER service."""
    manifest = load_ner_model_manifest()
    configured = os.getenv("NER_CONFIG__PROVIDER", manifest.production_provider).strip()
    if configured != manifest.production_provider:
        from app.ner.errors import NERConfigurationError

        raise NERConfigurationError(
            "NER_CONFIG__PROVIDER must match the approved production provider."
        )
    registry = create_production_registry(manifest)
    return MedicalNERService(registry.create(configured))

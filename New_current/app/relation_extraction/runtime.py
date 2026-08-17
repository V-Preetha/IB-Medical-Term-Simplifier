"""Composition root for the approved BioLinkBERT provider."""

from app.relation_extraction.config import RelationSettings
from app.relation_extraction.manifest import load_relation_model_manifest
from app.relation_extraction.providers import create_production_registry
from app.relation_extraction.service import RelationExtractionService


def create_relation_extraction_service() -> RelationExtractionService:
    manifest = load_relation_model_manifest()
    settings = RelationSettings.from_manifest(manifest)
    registry = create_production_registry(settings, manifest)
    return RelationExtractionService(registry.create(settings.provider))

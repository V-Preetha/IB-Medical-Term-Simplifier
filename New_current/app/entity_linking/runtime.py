"""Composition root for production entity linking."""

from app.entity_linking.config import EntityLinkingSettings
from app.entity_linking.manifest import load_entity_linking_manifest
from app.entity_linking.providers import create_production_registry
from app.entity_linking.service import EntityLinkingService


def create_entity_linking_service() -> EntityLinkingService:
    manifest = load_entity_linking_manifest()
    settings = EntityLinkingSettings.from_manifest(manifest)
    registry = create_production_registry(settings)
    return EntityLinkingService(registry.create(settings.provider))

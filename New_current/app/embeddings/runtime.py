"""Composition root for BioClinical ModernBERT embeddings."""

from app.embeddings.config import EmbeddingSettings
from app.embeddings.manifest import load_embedding_model_manifest
from app.embeddings.providers import create_production_registry
from app.embeddings.service import MedicalEmbeddingService


def create_embedding_service() -> MedicalEmbeddingService:
    manifest = load_embedding_model_manifest()
    settings = EmbeddingSettings.from_manifest(manifest)
    registry = create_production_registry(settings, manifest)
    return MedicalEmbeddingService(registry.create(settings.provider))

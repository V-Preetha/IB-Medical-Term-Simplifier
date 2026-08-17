import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.entity_linking.config import EntityLinkingSettings
from app.entity_linking.contracts import LinkerHealthStatus, SourceEntity
from app.entity_linking.errors import (
    EntityLinkingConfigurationError,
    UnsupportedEntityLinkingProviderError,
)
from app.entity_linking.manifest import load_entity_linking_manifest
from app.entity_linking.providers import EntityLinkingProviderRegistry
from app.entity_linking.service import EntityLinkingService
from app.main import create_app
from app.ner.service import MedicalNERService
from app.ocr.providers.runtime import ProviderContainer
from tests.entity_linking.fakes import FakeEntityLinkingProvider
from tests.ner.fakes import FakeNERProvider
from tests.ocr.fakes import FakeOCRProvider, FakePostProcessor


def _application():
    return create_app(
        provider_container=ProviderContainer(FakeOCRProvider(), FakePostProcessor()),
        ner_service=MedicalNERService(FakeNERProvider("biomedical-ner-all")),
        entity_linking_service=EntityLinkingService(FakeEntityLinkingProvider()),
    )


def test_manifest_is_explicitly_pending_and_runtime_fails_closed() -> None:
    manifest = load_entity_linking_manifest()
    assert manifest.provider == "scispacy-umls"
    assert manifest.terminology == "UMLS"
    assert manifest.approved is False
    settings = EntityLinkingSettings.from_manifest(manifest)
    with pytest.raises(EntityLinkingConfigurationError, match="awaits approved"):
        settings.validate()


def test_manifest_rejects_substituted_provider(tmp_path) -> None:
    manifest = load_entity_linking_manifest()
    contents = manifest.path.read_text(encoding="utf-8")
    contents = contents.replace('"provider": "scispacy-umls"', '"provider": "other"')
    candidate = tmp_path / "MODEL_MANIFEST.md"
    candidate.write_text(contents, encoding="utf-8")
    with pytest.raises(EntityLinkingConfigurationError, match="scispacy-umls"):
        load_entity_linking_manifest(candidate)


def test_registry_selection_and_duplicate_protection() -> None:
    registry = EntityLinkingProviderRegistry()
    registry.register("scispacy-umls", FakeEntityLinkingProvider)
    assert isinstance(registry.create("scispacy-umls"), FakeEntityLinkingProvider)
    with pytest.raises(EntityLinkingConfigurationError):
        registry.register("scispacy-umls", FakeEntityLinkingProvider)
    with pytest.raises(UnsupportedEntityLinkingProviderError):
        registry.create("unknown")


def test_service_preserves_concept_provenance() -> None:
    async def run():
        service = EntityLinkingService(FakeEntityLinkingProvider())
        await service.initialize()
        result = await service.process((SourceEntity("diabetes", "Disease", 0, 8, 0.98),))
        return service, result

    service, result = asyncio.run(run())
    assert result.links[0].selected_concept.concept_id == "C0011849"
    assert result.links[0].selected_concept.source_ontology == "UMLS"
    assert result.metadata["terminology_version"] == "2025AA"
    assert service.health().status is LinkerHealthStatus.READY


def test_api_health_models_linking_dashboard_and_openapi() -> None:
    with TestClient(_application()) as client:
        health = client.get("/api/v1/entity-linking/health")
        models = client.get("/api/v1/entity-linking/models")
        linked = client.post(
            "/api/v1/entity-linking",
            json={
                "entities": [
                    {
                        "text": "diabetes",
                        "label": "Disease",
                        "start": 0,
                        "end": 8,
                        "confidence": 0.98,
                    }
                ]
            },
        )
        dashboard = client.get("/entity-linking")
        openapi = client.get("/openapi.json").json()
    assert health.status_code == 200 and health.json()["status"] == "HEALTHY"
    assert models.json()["models"][0]["terminology_version"] == "2025AA"
    body = linked.json()
    assert body["schema_version"] == "entity-linking-response-v1"
    assert body["links"][0]["normalized_concept"]["concept_id"] == "C0011849"
    assert dashboard.status_code == 200 and "UMLS Entity Linking" in dashboard.text
    assert "/api/v1/entity-linking" in openapi["paths"]
    assert "/api/v1/relations" not in openapi["paths"]


def test_api_rejects_invalid_span_without_echoing_entity_text() -> None:
    with TestClient(_application()) as client:
        response = client.post(
            "/api/v1/entity-linking",
            json={
                "entities": [
                    {"text": "private", "label": "Disease", "start": 5, "end": 1, "confidence": 1}
                ]
            },
        )
    assert response.status_code == 422
    assert "private" not in json.dumps(response.json())

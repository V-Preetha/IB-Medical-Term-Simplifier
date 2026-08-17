import asyncio
import json

import pytest
import torch
from fastapi.testclient import TestClient

from app.embeddings.config import EmbeddingSettings
from app.embeddings.contracts import EmbeddingHealthStatus, EmbeddingInput
from app.embeddings.errors import (
    EmbeddingConfigurationError,
    UnsupportedEmbeddingProviderError,
)
from app.embeddings.manifest import load_embedding_model_manifest
from app.embeddings.providers import (
    EmbeddingProviderRegistry,
    _attention_mask_mean,
)
from app.embeddings.service import MedicalEmbeddingService
from app.main import create_app
from app.ner.service import MedicalNERService
from app.ocr.providers.runtime import ProviderContainer
from tests.embeddings.fakes import FakeEmbeddingProvider
from tests.ner.fakes import FakeNERProvider
from tests.ocr.fakes import FakeOCRProvider, FakePostProcessor


def _application():
    return create_app(
        provider_container=ProviderContainer(FakeOCRProvider(), FakePostProcessor()),
        ner_service=MedicalNERService(FakeNERProvider("biomedical-ner-all")),
        embedding_service=MedicalEmbeddingService(FakeEmbeddingProvider()),
    )


def test_manifest_records_pending_identity_without_guessing() -> None:
    manifest = load_embedding_model_manifest()
    assert manifest.model_name == "BioClinical ModernBERT"
    assert manifest.provider == "bioclinical-modernbert"
    assert manifest.approved is False
    with pytest.raises(EmbeddingConfigurationError, match="await approved"):
        EmbeddingSettings.from_manifest(manifest).validate(manifest)


def test_registry_is_instance_scoped_and_rejects_unknown_provider() -> None:
    registry = EmbeddingProviderRegistry()
    registry.register("bioclinical-modernbert", FakeEmbeddingProvider)
    assert isinstance(registry.create("bioclinical-modernbert"), FakeEmbeddingProvider)
    with pytest.raises(EmbeddingConfigurationError):
        registry.register("bioclinical-modernbert", FakeEmbeddingProvider)
    with pytest.raises(UnsupportedEmbeddingProviderError):
        registry.create("unknown")


def test_attention_mask_pooling_excludes_padding() -> None:
    hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = _attention_mask_mean(hidden, mask)
    assert pooled.tolist() == [[2.0, 4.0]]


def test_service_preserves_batch_order_and_model_metadata() -> None:
    async def run():
        service = MedicalEmbeddingService(FakeEmbeddingProvider())
        await service.initialize()
        result = await service.process(
            (
                EmbeddingInput("one", "Synthetic medical text one."),
                EmbeddingInput("two", "Synthetic medical text two."),
            )
        )
        return service, result

    service, result = asyncio.run(run())
    assert [item.input_id for item in result.embeddings] == ["one", "two"]
    assert result.embeddings[0].values == (0.6, 0.8)
    assert result.metadata["model_revision"] == "a" * 40
    assert service.health().status is EmbeddingHealthStatus.READY


def test_api_batch_health_models_dashboard_and_openapi() -> None:
    with TestClient(_application()) as client:
        response = client.post(
            "/api/v1/embeddings",
            json={
                "inputs": [
                    {"input_id": "one", "text": "Synthetic diabetes text."},
                    {"input_id": "two", "text": "Synthetic metformin text."},
                ]
            },
        )
        health = client.get("/api/v1/embeddings/health")
        models = client.get("/api/v1/embeddings/models")
        dashboard = client.get("/embeddings")
        openapi = client.get("/openapi.json").json()
    body = response.json()
    assert response.status_code == 200
    assert body["schema_version"] == "medical-embedding-response-v1"
    assert body["batch_size"] == 2
    assert body["embeddings"][0]["vector"] == [0.6, 0.8]
    assert body["confidence"] is None
    assert health.status_code == 200 and health.json()["status"] == "HEALTHY"
    assert models.json()["models"][0]["dimensions"] == 2
    assert dashboard.status_code == 200 and "Medical Embeddings" in dashboard.text
    assert "/api/v1/embeddings" in openapi["paths"]
    assert not any("qdrant" in path.casefold() for path in openapi["paths"])


def test_api_rejects_duplicate_ids_without_echoing_text() -> None:
    private_text = "synthetic private clinical text"
    with TestClient(_application()) as client:
        response = client.post(
            "/api/v1/embeddings",
            json={
                "inputs": [
                    {"input_id": "same", "text": private_text},
                    {"input_id": "same", "text": "other synthetic text"},
                ]
            },
        )
    assert response.status_code == 422
    assert private_text not in json.dumps(response.json())

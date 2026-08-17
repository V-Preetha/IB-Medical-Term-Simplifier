"""Production simplification contract, safety, API, and dashboard tests."""

from fastapi.testclient import TestClient

from app.main import create_app
from app.ner.service import MedicalNERService
from app.ocr.providers.runtime import ProviderContainer
from app.simplification.contracts import (
    BaseSimplificationProvider,
    MedicalTermExplanation,
    ProviderSimplificationResult,
    SimplificationLevelResult,
    SimplificationProviderMetadata,
)
from app.simplification.service import SimplificationService
from tests.ner.fakes import FakeNERProvider
from tests.ocr.fakes import FakeOCRProvider, FakePostProcessor


class FakeProvider(BaseSimplificationProvider):
    def __init__(self, *, introduce_value: bool = False) -> None:
        self.introduce_value = introduce_value

    async def initialize(self, *, strict: bool = True) -> None:
        del strict

    def simplify(self, text, entities, linked_concepts=()):
        del text, entities, linked_concepts
        report = "HbA1c was 7.2%." if not self.introduce_value else "HbA1c was 8.4%."
        levels = tuple(
            SimplificationLevelResult(
                level=name,
                simplified_report=report,
                medical_terms_explained=(
                    MedicalTermExplanation("HbA1c", "a measure of average blood sugar"),
                ),
                important_findings=(report,),
                suggested_questions_for_doctor=(
                    "What does the documented HbA1c result mean?",
                ),
            )
            for name in ("clinical", "general_public", "child_friendly")
        )
        return ProviderSimplificationResult(levels, 50, 80, 12.5)

    def metadata(self):
        return SimplificationProviderMetadata(
            "qwen3",
            "Qwen/Qwen3-0.6B",
            "c1899de289a04d12100db370d81485cdf75e47ca",
            "qwen-medical-simplification-v2",
            "cpu",
            True,
            "ready",
            {"local_files_only": True, "deterministic_generation": True},
            100.0,
        )

    async def shutdown(self) -> None:
        return None


def _app(provider: FakeProvider):
    return create_app(
        provider_container=ProviderContainer(FakeOCRProvider(), FakePostProcessor()),
        ner_service=MedicalNERService(FakeNERProvider("biomedical-ner-all")),
        simplification_service=SimplificationService(provider),
    )


def test_three_level_api_health_models_openapi_and_dashboard() -> None:
    with TestClient(_app(FakeProvider())) as client:
        response = client.post(
            "/api/v1/simplify",
            json={
                "text": "HbA1c was 7.2%.",
                "entities": [
                    {
                        "text": "HbA1c",
                        "label": "Laboratory Test",
                        "start": 0,
                        "end": 5,
                        "confidence": 0.98,
                    }
                ],
            },
        )
        health = client.get("/api/v1/simplify/health")
        models = client.get("/api/v1/simplify/models")
        dashboard = client.get("/simplify")
        script = client.get("/static/simplification_dashboard.js")
        schema = client.get("/openapi.json").json()

    assert response.status_code == 200
    body = response.json()
    assert [body[key]["level"] for key in ("clinical", "general_public", "child_friendly")] == [
        "clinical",
        "general_public",
        "child_friendly",
    ]
    assert body["clinical"]["original_report"] == "HbA1c was 7.2%."
    assert body["clinical"]["confidence"] == 1.0
    assert body["inference"]["prompt_version"] == "qwen-medical-simplification-v2"
    assert health.status_code == 200 and health.json()["status"] == "HEALTHY"
    assert models.json()["model_revision"] == "c1899de289a04d12100db370d81485cdf75e47ca"
    assert dashboard.status_code == 200 and "Medical Report Simplification" in dashboard.text
    assert "Level 3 - Child-Friendly" in script.text
    assert "/api/v1/simplify" in script.text
    assert "/api/v1/simplify" in schema["paths"]
    assert schema["paths"]["/api/v1/simplify"]["post"]["summary"]


def test_numeric_fabrication_fails_closed() -> None:
    with TestClient(_app(FakeProvider(introduce_value=True))) as client:
        response = client.post("/api/v1/simplify", json={"text": "HbA1c was 7.2%."})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsafe_simplification_output"

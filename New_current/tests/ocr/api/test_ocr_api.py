"""API contract tests for the converged OCR runtime."""

from fastapi.testclient import TestClient

from app.main import create_app
from app.ocr.providers.runtime import ProviderContainer
from tests.ocr.fakes import FakeOCRProvider, FakePostProcessor


def _application():
    container = ProviderContainer(
        ocr_provider=FakeOCRProvider(),
        postprocessor=FakePostProcessor(),
    )
    return create_app(provider_container=container)


def test_complete_ocr_endpoint_lifecycle() -> None:
    with TestClient(_application()) as client:
        created = client.post(
            "/api/v1/ocr",
            files={"file": ("synthetic.png", b"synthetic-image", "image/png")},
        )
        assert created.status_code == 201
        payload = created.json()
        assert payload["provider"] == "qwen3-vl"
        assert payload["normalized_text"].endswith("hemoglobin 6.5%")
        assert payload["confidence"] == 0.93
        assert payload["page_count"] == 1
        assert "content_sha256" not in payload["metadata"]

        request_id = payload["request_id"]
        status = client.get(f"/api/v1/ocr/status/{request_id}")
        assert status.status_code == 200
        assert status.json()["status"] == "completed"
        assert client.get(f"/api/v1/ocr/{request_id}").json() == payload
        assert client.delete(f"/api/v1/ocr/{request_id}").status_code == 204
        assert client.get(f"/api/v1/ocr/{request_id}").status_code == 404


def test_upload_validation_and_stable_error_envelope() -> None:
    with TestClient(_application(), raise_server_exceptions=False) as client:
        empty = client.post(
            "/api/v1/ocr",
            files={"file": ("empty.png", b"", "image/png")},
        )
        unsupported = client.post(
            "/api/v1/ocr",
            files={"file": ("unsafe.exe", b"synthetic", "application/octet-stream")},
        )
    assert empty.status_code == 400
    assert empty.json()["error"]["code"] == "invalid_upload"
    assert unsupported.status_code == 415
    assert unsupported.json()["error"]["code"] == "unsupported_document"


def test_health_models_dashboard_and_swagger_are_available() -> None:
    with TestClient(_application()) as client:
        assert client.get("/api/v1/health/live").json()["status"] == "live"
        assert client.get("/api/v1/health/ready").json()["status"] == "ready"
        health = client.get("/api/v1/ocr/health")
        models = client.get("/api/v1/ocr/models")
        recent = client.get("/api/v1/ocr/recent")
        logs = client.get("/api/v1/ocr/logs")
        dashboard = client.get("/")
        swagger = client.get("/docs")

    assert health.status_code == 200
    assert len(health.json()["providers"]) == 2
    assert models.status_code == 200
    assert {item["provider_name"] for item in models.json()["models"]} == {
        "qwen3-vl",
        "symspell",
    }
    assert recent.status_code == 200
    assert recent.json() == {"requests": []}
    assert logs.status_code == 200
    assert isinstance(logs.json()["records"], list)
    assert dashboard.status_code == 200
    assert "OCR Engineering Console" in dashboard.text
    assert swagger.status_code == 200


def test_openapi_documents_every_public_operation() -> None:
    schema = _application().openapi()
    required_paths = {
        "/api/v1/ocr",
        "/api/v1/ocr/status/{request_id}",
        "/api/v1/ocr/{request_id}",
        "/api/v1/ocr/health",
        "/api/v1/ocr/models",
        "/api/v1/health/live",
        "/api/v1/health/ready",
    }
    assert required_paths <= set(schema["paths"])
    for path in required_paths:
        for operation in schema["paths"][path].values():
            assert operation["summary"]
            assert operation["description"]
            assert operation["responses"]

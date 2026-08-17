"""Tests for the Stage 1 API skeleton."""

from fastapi.testclient import TestClient

from app.main import create_app


def test_health_check_returns_service_metadata() -> None:
    """Health route should expose only client-safe service metadata."""
    client = TestClient(create_app())

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service_name": "AI Medical Report Simplifier",
        "environment": "local",
        "version": "0.1.0",
    }


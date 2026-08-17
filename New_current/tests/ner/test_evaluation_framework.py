import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.ner.contracts import NormalizedEntity
from app.ner.errors import NERConfigurationError, UnsupportedNERModelError
from app.ner.manifest import load_ner_model_manifest
from app.ner.providers import (
    NERProviderRegistry,
    _decode_token_entities,
    _window_owns_entity,
    create_production_registry,
)
from app.ner.runtime import create_ner_service
from app.ner.service import MedicalNERService
from app.ocr.providers.runtime import ProviderContainer
from benchmarks.ner.service import NERBenchmarkService
from tests.ner.fakes import FakeNERProvider
from tests.ocr.fakes import FakeOCRProvider, FakePostProcessor


def _benchmark_service() -> NERBenchmarkService:
    registry = NERProviderRegistry()
    for name in ("openmed-gliner", "biomedical-ner-all", "modernbert-biomedical-ner"):
        registry.register(name, lambda candidate=name: FakeNERProvider(candidate))
    return NERBenchmarkService(registry)


def _production_service() -> MedicalNERService:
    return MedicalNERService(FakeNERProvider("biomedical-ner-all"))


def _application():
    return create_app(
        provider_container=ProviderContainer(FakeOCRProvider(), FakePostProcessor()),
        ner_service=_production_service(),
    )


def test_manifest_defines_exactly_three_approved_candidates() -> None:
    manifest = load_ner_model_manifest()
    assert set(manifest.candidates) == {
        "openmed-gliner",
        "biomedical-ner-all",
        "modernbert-biomedical-ner",
    }
    assert all(item.approved for item in manifest.candidates.values())
    assert all(len(item.pinned_revision) == 40 for item in manifest.candidates.values())
    assert manifest.production_provider == "biomedical-ner-all"


def test_registry_rejects_unknown_candidate() -> None:
    with pytest.raises(UnsupportedNERModelError):
        NERProviderRegistry().create("not-approved")


def test_production_registry_contains_only_approved_winner() -> None:
    manifest = load_ner_model_manifest()
    registry = create_production_registry(manifest)
    assert registry.names() == ("biomedical-ner-all",)
    provider = registry.create("biomedical-ner-all")
    entities, ignored = _decode_token_entities(
        "diabetes metformin",
        [[0, 8], [9, 12], [12, 18]],
        [1, 2, 3],
        [0.99, 0.98, 0.97],
        {
            1: "B-Disease_disorder",
            2: "B-Medication",
            3: "I-Medication",
        },
        provider._label_aliases,
        0.5,
    )
    assert ignored == set()
    assert [(entity.text, entity.label) for entity in entities] == [
        ("diabetes", "Disease"),
        ("metformin", "Medication"),
    ]


def test_production_provider_selection_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NER_CONFIG__PROVIDER", "openmed-gliner")
    with pytest.raises(NERConfigurationError):
        create_ner_service()


def test_overlapping_windows_assign_each_entity_to_one_owner() -> None:
    bounds = [(0, 100), (70, 170)]
    first_overlap_entity = NormalizedEntity("entity", "Disease", 77, 83, 0.9)
    second_overlap_entity = NormalizedEntity("entity", "Disease", 87, 93, 0.9)
    assert _window_owns_entity(first_overlap_entity, 0, bounds)
    assert not _window_owns_entity(first_overlap_entity, 1, bounds)
    assert not _window_owns_entity(second_overlap_entity, 0, bounds)
    assert _window_owns_entity(second_overlap_entity, 1, bounds)


def test_token_output_is_normalized_and_bio_spans_are_merged() -> None:
    text = "severe diabetes"
    entities, ignored = _decode_token_entities(
        text,
        [[0, 6], [7, 15]],
        [1, 2],
        [0.8, 0.9],
        {1: "B-DISEASE", 2: "I-DISEASE"},
        {"disease": "Disease"},
        0.5,
    )
    assert ignored == set()
    assert entities[0].text == "severe diabetes"
    assert entities[0].label == "Disease"
    assert entities[0].confidence == pytest.approx(0.85)


def test_token_output_trims_tokenizer_whitespace_offsets() -> None:
    text = "severe diabetes"
    entities, ignored = _decode_token_entities(
        text,
        [[0, 6], [6, 15]],
        [1, 2],
        [0.8, 0.9],
        {1: "B-DISEASE", 2: "I-DISEASE"},
        {"disease": "Disease"},
        0.5,
    )
    assert ignored == set()
    assert entities[0].text == "severe diabetes"
    assert (entities[0].start, entities[0].end) == (0, 15)


def test_service_calculates_exact_span_metrics() -> None:
    text = "Diabetes requires monitoring."
    reference = (NormalizedEntity("Diabetes", "Disease", 0, 8, 1.0),)
    service = _benchmark_service()
    result = asyncio.run(service.benchmark("openmed-gliner", text, reference))
    assert result.entities == (NormalizedEntity("Diabetes", "Disease", 0, 8, 0.95),)
    assert result.metrics.precision == 1.0
    assert result.metrics.recall == 1.0
    assert result.metrics.f1_score == 1.0
    assert result.metrics.entity_level_accuracy == 1.0
    assert result.metrics.false_positives == 0
    assert result.metrics.false_negatives == 0
    assert result.metrics.inference_latency_ms >= 0
    assert result.metrics.peak_ram_mb > 0
    assert result.metrics.tokens_per_second is not None
    asyncio.run(service.shutdown())


def test_production_api_models_health_dashboard_and_swagger() -> None:
    with TestClient(_application()) as client:
        models = client.get("/api/v1/ner/models")
        health = client.get("/api/v1/ner/health")
        dashboard = client.get("/ner")
        inference = client.post(
            "/api/v1/ner",
            json={"text": "Diabetes requires monitoring."},
        )
        swagger = client.get("/docs")
    assert models.status_code == 200
    assert len(models.json()["models"]) == 1
    assert models.json()["models"][0]["provider_name"] == "biomedical-ner-all"
    assert health.status_code == 200
    assert health.json()["status"] == "HEALTHY"
    assert dashboard.status_code == 200
    assert "Medical NER Console" in dashboard.text
    assert "JSON response" in dashboard.text
    assert inference.status_code == 200
    assert inference.json()["entities"][0]["label"] == "Disease"
    assert inference.json()["model_revision"] == "a" * 40
    assert inference.json()["confidence"] == 0.95
    assert inference.json()["processing_time_ms"] >= 0
    assert swagger.status_code == 200


def test_empty_ner_input_is_rejected() -> None:
    with TestClient(_application(), raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/ner",
            json={"text": ""},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "input_validation_error"
    assert response.json()["request_id"]


def test_openapi_documents_only_production_ner_endpoints() -> None:
    schema = _application().openapi()
    expected = {
        "/api/v1/ner": "post",
        "/api/v1/ner/models": "get",
        "/api/v1/ner/health": "get",
    }
    for path, method in expected.items():
        operation = schema["paths"][path][method]
        assert operation["summary"]
        assert operation["description"]
        assert operation["responses"]
    assert "/api/v1/ner/benchmark" not in schema["paths"]


def test_benchmark_runner_generates_evaluation_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarks.ner import run_benchmark

    benchmark_root = Path(run_benchmark.__file__).parent
    monkeypatch.setattr(run_benchmark, "create_ner_benchmark_service", _benchmark_service)
    monkeypatch.setattr(
        run_benchmark,
        "_model_inventory",
        lambda entry, metadata: {"repository_id": entry.repository_id},
    )
    output = tmp_path / "reports"
    payload = asyncio.run(
        run_benchmark.run(
            benchmark_root / "dataset_template.jsonl",
            output,
            benchmark_root / "evaluation_config.json",
        )
    )
    assert payload["winner"] is None
    assert {item["status"] for item in payload["candidates"].values()} == {"PASS"}
    assert payload["recommended_candidate"] in payload["candidates"]
    assert (output / "ner_benchmark_report.json").is_file()
    assert (output / "ner_benchmark_metrics.csv").is_file()
    assert (output / "ner_benchmark_per_entity.csv").is_file()
    assert (output / "ner_benchmark_leaderboard.csv").is_file()
    assert (output / "ner_benchmark_report.md").is_file()

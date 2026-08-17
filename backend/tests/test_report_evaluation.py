"""Tests for Stage 10 report evaluation."""

from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.evaluation.report_evaluation import (
    BERTScoreMetrics,
    EvaluationError,
    EvaluationService,
)
from app.fusion.medical_fusion import FusedMedicalTerm
from app.main import create_app


class FakeBERTScoreBackend:
    """Deterministic BERTScore backend for tests."""

    def score(self, *, reference: str, candidate: str) -> BERTScoreMetrics:
        """Return fixed BERTScore metrics."""
        return BERTScoreMetrics(precision=0.91, recall=0.92, f1=0.915)


class FakeSemanticBackend:
    """Deterministic semantic similarity backend for tests."""

    def similarity(self, *, reference: str, candidate: str) -> float:
        """Return fixed cosine similarity."""
        return 0.88


def _fused_term() -> FusedMedicalTerm:
    """Build a fused term for evaluation tests."""
    return FusedMedicalTerm(
        term="Hypertension",
        difficulty=0.826,
        meaning="High blood pressure.",
        context="A chronic condition where blood pressure remains higher than normal.",
        confidence=0.9393,
        entity_type="disease",
        section_type="diagnosis",
        modernbert_confidence=0.92,
        semantic_confidence=0.945,
        ambiguity_resolution="Resolved as chronic disease.",
        modernbert_embedding=[0.6, 0.8],
        semantic_embedding=[1.0, 0.0],
        matched_concept="hypertension",
    )


def _payload(simplified_report: str) -> dict[str, object]:
    """Build API payload for evaluation tests."""
    return {
        "reference_text": "Hypertension means high blood pressure.",
        "simplified_report": simplified_report,
        "fused_terms": [
            {
                "term": "Hypertension",
                "difficulty": 0.826,
                "meaning": "High blood pressure.",
                "context": "A chronic condition where blood pressure remains higher than normal.",
                "confidence": 0.9393,
                "entity_type": "disease",
                "section_type": "diagnosis",
                "modernbert_confidence": 0.92,
                "semantic_confidence": 0.945,
                "ambiguity_resolution": "Resolved as chronic disease.",
                "modernbert_embedding": [0.6, 0.8],
                "semantic_embedding": [1.0, 0.0],
                "matched_concept": "hypertension",
            }
        ],
    }


def test_evaluation_computes_all_required_metrics() -> None:
    """Evaluation should include BERTScore, semantic, readability, and consistency."""
    service = EvaluationService(
        Settings(),
        bertscore_backend=FakeBERTScoreBackend(),
        semantic_backend=FakeSemanticBackend(),
    )

    result = service.evaluate(
        reference_text="Hypertension means high blood pressure.",
        simplified_report="Hypertension means high blood pressure.",
        fused_terms=[_fused_term()],
    )

    assert result.bertscore.f1 == 0.915
    assert result.semantic_similarity == 0.88
    assert result.readability.flesch_kincaid_grade_level == 7.6
    assert result.readability.flesch_reading_ease == 49.48
    assert result.medical_consistency.score == 1.0
    assert result.warnings == []


def test_evaluation_flags_medical_consistency_failures() -> None:
    """Missing terms and unsupported numbers should reduce medical consistency."""
    service = EvaluationService(
        Settings(),
        bertscore_backend=FakeBERTScoreBackend(),
        semantic_backend=FakeSemanticBackend(),
    )

    result = service.evaluate(
        reference_text="Hypertension means high blood pressure.",
        simplified_report="This is blood pressure of 120.",
        fused_terms=[_fused_term()],
    )

    assert result.medical_consistency.score < 1.0
    assert result.medical_consistency.terms_preserved is False
    assert result.medical_consistency.unsupported_numbers == ["120"]
    assert result.warnings == [
        "Medical consistency checks found missing or unsupported content."
    ]


def test_evaluation_requires_inputs() -> None:
    """Evaluation should reject missing source fields."""
    service = EvaluationService(
        Settings(),
        bertscore_backend=FakeBERTScoreBackend(),
        semantic_backend=FakeSemanticBackend(),
    )

    try:
        service.evaluate(
            reference_text="",
            simplified_report="text",
            fused_terms=[_fused_term()],
        )
    except EvaluationError as exc:
        assert "Reference text" in str(exc)
    else:
        raise AssertionError("Expected EvaluationError")


def test_evaluation_endpoint_returns_metrics_json() -> None:
    """The API should expose Stage 10 metrics."""
    app = create_app()
    client = TestClient(app)

    from app.api.routes.simplify import get_evaluation_service

    app.dependency_overrides[get_evaluation_service] = lambda: EvaluationService(
        Settings(),
        bertscore_backend=FakeBERTScoreBackend(),
        semantic_backend=FakeSemanticBackend(),
    )

    response = client.post(
        "/api/v1/reports/evaluate",
        json=_payload("Hypertension means high blood pressure."),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["bertscore"] == {
        "precision": 0.91,
        "recall": 0.92,
        "f1": 0.915,
    }
    assert payload["semantic_similarity"] == 0.88
    assert payload["medical_consistency"]["score"] == 1.0

    app.dependency_overrides.clear()

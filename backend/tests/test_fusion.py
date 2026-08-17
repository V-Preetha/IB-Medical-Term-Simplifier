"""Tests for Stage 7 fusion layer."""

from fastapi.testclient import TestClient

from app.fusion.medical_fusion import FusionError, FusionService
from app.main import create_app
from app.services.clinical_context import SemanticInterpretation
from app.services.entity_recognition import MedicalEntityType
from app.services.modernbert import DifficultTerm


def _difficult_term(term: str = "Hypertension") -> DifficultTerm:
    """Build a ModernBERT difficult term for tests."""
    return DifficultTerm(
        term=term,
        difficulty=0.826,
        embedding=[0.6, 0.8],
        confidence=0.92,
        entity_type=MedicalEntityType.DISEASE,
        section_type="diagnosis",
        context=f"Diagnosis: {term}",
    )


def _interpretation(term: str = "Hypertension") -> SemanticInterpretation:
    """Build a semantic interpretation for tests."""
    return SemanticInterpretation(
        term=term,
        meaning="High blood pressure.",
        context="A chronic condition where blood pressure remains higher than normal.",
        ambiguity_resolution="Resolved as the chronic disease.",
        confidence=0.945,
        entity_type=MedicalEntityType.DISEASE,
        section_type="diagnosis",
        semantic_embedding=[1.0, 0.0],
        matched_concept="hypertension",
    )


def test_fusion_combines_modernbert_and_semantic_outputs() -> None:
    """Fusion should preserve both model outputs and compute confidence."""
    service = FusionService()

    result = service.fuse(
        difficult_terms=[_difficult_term()],
        semantic_interpretations=[_interpretation()],
    )

    assert result.term_count == 1
    fused = result.terms[0]
    assert fused.term == "Hypertension"
    assert fused.difficulty == 0.826
    assert fused.meaning == "High blood pressure."
    assert fused.context == "A chronic condition where blood pressure remains higher than normal."
    assert fused.confidence == 0.9393
    assert fused.modernbert_confidence == 0.92
    assert fused.semantic_confidence == 0.945
    assert fused.modernbert_embedding == [0.6, 0.8]
    assert fused.semantic_embedding == [1.0, 0.0]
    assert result.warnings == []


def test_fusion_warns_about_unmatched_semantic_outputs() -> None:
    """Unmatched semantic terms should be surfaced as warnings."""
    service = FusionService()

    result = service.fuse(
        difficult_terms=[_difficult_term("Hypertension")],
        semantic_interpretations=[
            _interpretation("Hypertension"),
            _interpretation("Diabetes"),
        ],
    )

    assert result.term_count == 1
    assert result.warnings == [
        "Semantic interpretation for 'Diabetes' had no ModernBERT difficult-term match."
    ]


def test_fusion_requires_overlap() -> None:
    """No shared terms should fail instead of producing misleading output."""
    service = FusionService()

    try:
        service.fuse(
            difficult_terms=[_difficult_term("Hypertension")],
            semantic_interpretations=[_interpretation("Diabetes")],
        )
    except FusionError as exc:
        assert "No overlapping" in str(exc)
    else:
        raise AssertionError("Expected FusionError")


def test_fusion_endpoint_returns_structured_representation() -> None:
    """The API should return the unified Stage 7 representation."""
    client = TestClient(create_app())

    response = client.post(
        "/api/v1/reports/fusion",
        json={
            "difficult_terms": [
                {
                    "term": "Hypertension",
                    "difficulty": 0.826,
                    "embedding": [0.6, 0.8],
                    "confidence": 0.92,
                    "entity_type": "disease",
                    "section_type": "diagnosis",
                    "context": "Diagnosis: Hypertension",
                }
            ],
            "semantic_interpretations": [
                {
                    "term": "Hypertension",
                    "meaning": "High blood pressure.",
                    "context": "A chronic condition where blood pressure remains higher than normal.",
                    "ambiguity_resolution": "Resolved as the chronic disease.",
                    "confidence": 0.945,
                    "entity_type": "disease",
                    "section_type": "diagnosis",
                    "semantic_embedding": [1.0, 0.0],
                    "matched_concept": "hypertension",
                }
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["terms"][0]["term"] == "Hypertension"
    assert payload["terms"][0]["meaning"] == "High blood pressure."
    assert payload["terms"][0]["confidence"] == 0.9393
    assert payload["algorithm_version"] == "weighted-key-match-v1"

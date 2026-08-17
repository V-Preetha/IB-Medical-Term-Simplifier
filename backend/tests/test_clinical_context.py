"""Tests for Stage 6 BioClinicalBERT/OpenMed semantic context processing."""

from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.main import create_app
from app.services.clinical_context import (
    ClinicalContextError,
    ClinicalContextService,
)
from app.services.entity_recognition import MedicalEntity, MedicalEntityType
from app.services.section_segmentation import ClinicalSectionType


class FakeSemanticBackend:
    """Deterministic semantic embedding backend for tests."""

    model_name = "fake-bioclinicalbert"

    def embed(self, text: str) -> list[float]:
        """Return stable vectors that make exact clinical concepts closest."""
        text = text.lower()
        if "hypertension" in text:
            return [1.0, 0.0, 0.0]
        if "diabetes" in text:
            return [0.0, 0.5, 0.5]
        if "lisinopril" in text:
            return [0.0, 1.0, 0.0]
        if "ldl" in text:
            return [0.0, 0.0, 1.0]
        return [0.1, 0.1, 0.1]


def _entity(
    text: str,
    entity_type: MedicalEntityType,
    section_type: ClinicalSectionType,
) -> MedicalEntity:
    """Build a medical entity for semantic tests."""
    return MedicalEntity(
        text=text,
        entity_type=entity_type,
        section_type=section_type,
        section_title=section_type.value.title(),
        start_char=0,
        end_char=len(text),
        confidence=0.9,
        source_label=None,
    )


def test_clinical_context_resolves_meaning_and_ambiguity() -> None:
    """Known clinical terms should map to semantic meaning and context."""
    service = ClinicalContextService(
        Settings(semantic_match_threshold=0.35),
        embedding_backend=FakeSemanticBackend(),
    )

    result = service.interpret(
        [
            _entity(
                "Hypertension",
                MedicalEntityType.DISEASE,
                ClinicalSectionType.DIAGNOSIS,
            )
        ]
    )

    assert result.interpretation_count == 1
    interpretation = result.interpretations[0]
    assert interpretation.term == "Hypertension"
    assert interpretation.meaning == "High blood pressure."
    assert interpretation.matched_concept == "hypertension"
    assert "chronic condition" in interpretation.context
    assert "chronic disease" in interpretation.ambiguity_resolution
    assert interpretation.confidence > 0.9


def test_clinical_context_uses_fallback_for_unknown_terms() -> None:
    """Unknown terms should preserve type and section context with a warning."""
    service = ClinicalContextService(
        Settings(semantic_match_threshold=0.99),
        embedding_backend=FakeSemanticBackend(),
    )

    result = service.interpret(
        [_entity("RareTerm", MedicalEntityType.DISEASE, ClinicalSectionType.HISTORY)]
    )

    assert result.interpretations[0].meaning == "A medical condition or diagnosis."
    assert result.interpretations[0].matched_concept is None
    assert result.warnings == ["No glossary concept met semantic threshold for 'RareTerm'."]


def test_clinical_context_requires_entities() -> None:
    """Stage 6 requires Stage 4 medical entities."""
    service = ClinicalContextService(Settings(), embedding_backend=FakeSemanticBackend())

    try:
        service.interpret([])
    except ClinicalContextError as exc:
        assert "At least one" in str(exc)
    else:
        raise AssertionError("Expected ClinicalContextError")


def test_clinical_context_endpoint_uses_dependency_injected_service() -> None:
    """The API should expose semantic interpretations as structured JSON."""
    app = create_app()
    client = TestClient(app)

    from app.api.routes.simplify import get_clinical_context_service

    app.dependency_overrides[get_clinical_context_service] = lambda: ClinicalContextService(
        Settings(semantic_match_threshold=0.35),
        embedding_backend=FakeSemanticBackend(),
    )

    response = client.post(
        "/api/v1/reports/clinical-context",
        json={
            "entities": [
                {
                    "text": "Hypertension",
                    "entity_type": "disease",
                    "section_type": "diagnosis",
                    "section_title": "Diagnosis",
                    "start_char": 0,
                    "end_char": 12,
                    "confidence": 0.9,
                    "source_label": "DISEASE",
                }
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()["interpretations"][0]["meaning"] == "High blood pressure."
    assert response.json()["interpretations"][0]["matched_concept"] == "hypertension"
    assert response.json()["model_name"] == "fake-bioclinicalbert"

    app.dependency_overrides.clear()

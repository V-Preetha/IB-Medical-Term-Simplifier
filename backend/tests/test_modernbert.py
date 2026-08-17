"""Tests for Stage 5 ModernBERT difficult-term processing."""

from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.main import create_app
from app.services.entity_recognition import MedicalEntity, MedicalEntityType
from app.services.modernbert import (
    ModernBERTProcessingError,
    ModernBERTProcessingService,
)
from app.services.section_segmentation import ClinicalSectionType


class FakeEmbeddingBackend:
    """Deterministic ModernBERT embedding backend test double."""

    model_name = "fake-modernbert"

    def embed(self, *, term: str, context: str) -> list[float]:
        """Return a stable non-zero embedding."""
        return [0.6, 0.8]


def _entity(
    text: str,
    entity_type: MedicalEntityType,
    section_type: ClinicalSectionType,
    confidence: float = 0.9,
) -> MedicalEntity:
    """Build a medical entity for tests."""
    return MedicalEntity(
        text=text,
        entity_type=entity_type,
        section_type=section_type,
        section_title=section_type.value.title(),
        start_char=0,
        end_char=len(text),
        confidence=confidence,
        source_label=None,
    )


def test_modernbert_identifies_difficult_terms_with_embeddings() -> None:
    """Difficult entities should include difficulty, embedding, and confidence."""
    service = ModernBERTProcessingService(
        Settings(difficult_term_threshold=0.55),
        embedding_backend=FakeEmbeddingBackend(),
    )

    result = service.process(
        [
            _entity(
                "Hypertension",
                MedicalEntityType.DISEASE,
                ClinicalSectionType.DIAGNOSIS,
            ),
            _entity("pain", MedicalEntityType.SYMPTOM, ClinicalSectionType.FINDINGS),
        ]
    )

    assert result.term_count == 1
    assert result.model_name == "fake-modernbert"
    assert result.terms[0].term == "Hypertension"
    assert result.terms[0].difficulty >= 0.55
    assert result.terms[0].embedding == [0.6, 0.8]
    assert result.terms[0].confidence == 0.92


def test_modernbert_requires_entities() -> None:
    """Stage 5 requires Stage 4 entities as input."""
    service = ModernBERTProcessingService(
        Settings(),
        embedding_backend=FakeEmbeddingBackend(),
    )

    try:
        service.process([])
    except ModernBERTProcessingError as exc:
        assert "At least one" in str(exc)
    else:
        raise AssertionError("Expected ModernBERTProcessingError")


def test_modernbert_endpoint_uses_dependency_injected_processor() -> None:
    """The API should expose difficult-term detection as structured JSON."""
    app = create_app()
    client = TestClient(app)

    from app.api.routes.simplify import get_modernbert_processor

    app.dependency_overrides[get_modernbert_processor] = lambda: ModernBERTProcessingService(
        Settings(difficult_term_threshold=0.55),
        embedding_backend=FakeEmbeddingBackend(),
    )

    response = client.post(
        "/api/v1/reports/modernbert/difficult-terms",
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
    assert response.json() == {
        "terms": [
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
        "term_count": 1,
        "model_name": "fake-modernbert",
        "warnings": [],
    }

    app.dependency_overrides.clear()

"""Tests for Stage 4 SciSpaCy-backed medical entity recognition."""

from dataclasses import dataclass

from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.main import create_app
from app.services.entity_recognition import (
    EntityRecognitionError,
    MedicalEntityRecognitionService,
    MedicalEntityType,
)
from app.services.section_segmentation import ClinicalSection, ClinicalSectionType


@dataclass(frozen=True)
class FakeSpan:
    """Minimal spaCy Span test double."""

    text: str
    start_char: int
    end_char: int
    label_: str


@dataclass(frozen=True)
class FakeDoc:
    """Minimal spaCy Doc test double."""

    ents: list[FakeSpan]


class FakeNlp:
    """Deterministic SciSpaCy-like test double."""

    def __call__(self, text: str) -> FakeDoc:
        """Return entities based on the supplied text."""
        spans: list[FakeSpan] = []
        for term, label in {
            "Hypertension": "DISEASE",
            "chest pain": "ENTITY",
            "Lisinopril": "CHEMICAL",
            "LDL": "ENTITY",
        }.items():
            start = text.find(term)
            if start >= 0:
                spans.append(FakeSpan(term, start, start + len(term), label))
        return FakeDoc(spans)


def test_entity_recognizer_extracts_required_categories() -> None:
    """SciSpaCy spans should be classified into required medical buckets."""
    recognizer = MedicalEntityRecognitionService(Settings(), nlp_model=FakeNlp())

    result = recognizer.extract(
        [
            ClinicalSection(
                section_type=ClinicalSectionType.DIAGNOSIS,
                title="Diagnosis",
                content="Hypertension with chest pain",
                order=0,
                confidence=0.95,
            ),
            ClinicalSection(
                section_type=ClinicalSectionType.MEDICATIONS,
                title="Medications",
                content="Lisinopril 10 mg daily",
                order=1,
                confidence=0.95,
            ),
            ClinicalSection(
                section_type=ClinicalSectionType.LAB_RESULTS,
                title="Lab Results",
                content="LDL 160 mg/dL",
                order=2,
                confidence=0.95,
            ),
        ]
    )

    assert result.entity_count == 4
    assert [(entity.text, entity.entity_type) for entity in result.entities] == [
        ("Hypertension", MedicalEntityType.DISEASE),
        ("chest pain", MedicalEntityType.SYMPTOM),
        ("Lisinopril", MedicalEntityType.DRUG),
        ("LDL", MedicalEntityType.LAB_TEST),
    ]


def test_entity_recognizer_requires_sections() -> None:
    """Recognition needs Stage 3 segmented sections."""
    recognizer = MedicalEntityRecognitionService(Settings(), nlp_model=FakeNlp())

    try:
        recognizer.extract([])
    except EntityRecognitionError as exc:
        assert "At least one" in str(exc)
    else:
        raise AssertionError("Expected EntityRecognitionError")


def test_entity_endpoint_uses_dependency_injected_recognizer() -> None:
    """The API should return structured entity JSON."""
    app = create_app()
    client = TestClient(app)

    from app.api.routes.simplify import get_entity_recognizer

    app.dependency_overrides[get_entity_recognizer] = lambda: MedicalEntityRecognitionService(
        Settings(), nlp_model=FakeNlp()
    )

    response = client.post(
        "/api/v1/reports/entities",
        json={
            "sections": [
                {
                    "section_type": "diagnosis",
                    "title": "Diagnosis",
                    "content": "Hypertension",
                    "order": 0,
                    "confidence": 0.95,
                }
            ]
        },
    )

    assert response.status_code == 200
    assert response.json() == {
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
        ],
        "entity_count": 1,
        "warnings": [],
    }

    app.dependency_overrides.clear()

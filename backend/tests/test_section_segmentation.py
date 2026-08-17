"""Tests for Stage 3 clinical section segmentation."""

from fastapi.testclient import TestClient

from app.main import create_app
from app.services.section_segmentation import (
    ClinicalSectionType,
    SectionSegmentationError,
    SectionSegmentationService,
)


def test_segmenter_splits_known_clinical_sections() -> None:
    """Known headings should become normalized ordered sections."""
    segmenter = SectionSegmentationService()

    segmented_report = segmenter.segment(
        """
        Patient: Jane Doe

        Diagnosis: Hypertension
        Lab Results: LDL 160 mg/dL
        Medications: Lisinopril 10 mg daily
        Recommendations: Follow up in 2 weeks
        """
    )

    assert segmented_report.section_count == 5
    assert [section.section_type for section in segmented_report.sections] == [
        ClinicalSectionType.OTHER,
        ClinicalSectionType.DIAGNOSIS,
        ClinicalSectionType.LAB_RESULTS,
        ClinicalSectionType.MEDICATIONS,
        ClinicalSectionType.RECOMMENDATIONS,
    ]
    assert segmented_report.sections[1].content == "Hypertension"
    assert segmented_report.sections[3].content == "Lisinopril 10 mg daily"


def test_segmenter_maps_heading_synonyms() -> None:
    """Common report heading synonyms should map to canonical section types."""
    segmenter = SectionSegmentationService()

    segmented_report = segmenter.segment(
        "Past Medical History: Diabetes\nPlan: Continue metformin"
    )

    assert [section.section_type for section in segmented_report.sections] == [
        ClinicalSectionType.HISTORY,
        ClinicalSectionType.RECOMMENDATIONS,
    ]


def test_segmenter_returns_other_when_no_headings_are_found() -> None:
    """Text without known headings should not be discarded."""
    segmenter = SectionSegmentationService()

    segmented_report = segmenter.segment("Patient has mild cough and fever.")

    assert segmented_report.section_count == 1
    assert segmented_report.sections[0].section_type is ClinicalSectionType.OTHER
    assert segmented_report.warnings == [
        "No known clinical section headings were detected."
    ]


def test_segmenter_rejects_empty_text() -> None:
    """Blank input should fail with a segmentation-specific error."""
    segmenter = SectionSegmentationService()

    try:
        segmenter.segment("   \n\t  ")
    except SectionSegmentationError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("Expected SectionSegmentationError")


def test_segment_endpoint_returns_structured_json() -> None:
    """The Stage 3 endpoint should expose section segmentation."""
    client = TestClient(create_app())

    response = client.post(
        "/api/v1/reports/segment",
        json={"text": "Findings: No acute disease\nImpression: Normal study"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "sections": [
            {
                "section_type": "findings",
                "title": "Findings",
                "content": "No acute disease",
                "order": 0,
                "confidence": 0.95,
            },
            {
                "section_type": "impression",
                "title": "Impression",
                "content": "Normal study",
                "order": 1,
                "confidence": 0.95,
            },
        ],
        "section_count": 2,
        "warnings": [],
    }

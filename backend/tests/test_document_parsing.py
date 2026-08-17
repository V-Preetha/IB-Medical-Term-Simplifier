"""Tests for Stage 2 document parsing."""

from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.main import create_app
from app.services.document_parsing import (
    DocumentParsingError,
    DocumentParsingService,
    ExtractionMethod,
    SourceType,
    clean_extracted_text,
)


def test_plain_text_parser_cleans_extracted_text() -> None:
    """Plain text parsing should normalize whitespace without changing values."""
    parser = DocumentParsingService(Settings())

    parsed_document = parser.parse_text(" BP:   140/90 \n\n\n Diagnosis:  HTN ")

    assert parsed_document.text == "BP: 140/90\n\nDiagnosis: HTN"
    assert parsed_document.source_type is SourceType.TEXT
    assert parsed_document.extraction_method is ExtractionMethod.TEXT_DIRECT
    assert parsed_document.ocr_applied is False


def test_empty_text_parser_raises_domain_error() -> None:
    """Blank text should fail with a parser-specific exception."""
    parser = DocumentParsingService(Settings())

    try:
        parser.parse_text("   \n\t  ")
    except DocumentParsingError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("Expected DocumentParsingError")


def test_text_file_parser_detects_text_from_content_type() -> None:
    """Uploaded text files should be decoded and cleaned."""
    parser = DocumentParsingService(Settings())

    parsed_document = parser.parse_file(
        filename="report.unknown",
        content=b" Findings:   Normal ",
        content_type="text/plain",
    )

    assert parsed_document.text == "Findings: Normal"
    assert parsed_document.source_type is SourceType.TEXT


def test_clean_extracted_text_preserves_line_structure() -> None:
    """Cleaning should preserve paragraph breaks for downstream segmentation."""
    assert clean_extracted_text("A   B\n\n\n C\tD") == "A B\n\nC D"


def test_extract_endpoint_accepts_text_form_data() -> None:
    """The Stage 2 endpoint should return parsed text metadata."""
    client = TestClient(create_app())

    response = client.post(
        "/api/v1/reports/extract",
        data={"text": "Diagnosis:   Hypertension"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "text": "Diagnosis: Hypertension",
        "source_type": "text",
        "extraction_method": "text_direct",
        "ocr_applied": False,
        "page_count": None,
        "warnings": [],
    }


def test_extract_endpoint_rejects_ambiguous_input() -> None:
    """Clients should send exactly one input source."""
    client = TestClient(create_app())

    response = client.post("/api/v1/reports/extract")

    assert response.status_code == 400
    assert response.json()["detail"] == "Provide exactly one input: either text or file."

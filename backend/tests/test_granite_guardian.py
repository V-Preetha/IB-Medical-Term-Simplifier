"""Tests for Stage 9 Granite Guardian validation."""

from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.fusion.medical_fusion import FusedMedicalTerm
from app.main import create_app
from app.services.granite_guardian import (
    GraniteGuardianValidationService,
    GuardianAssessment,
    ValidationAction,
    ValidationError,
)


class FakeGuardianBackend:
    """Deterministic IBM Granite Guardian backend test double."""

    model_name = "fake-granite-guardian"

    def __init__(
        self,
        *,
        hallucination_risk: float = 0.1,
        factual_consistency_risk: float = 0.1,
        unsafe_content_risk: float = 0.1,
        terminology_risk: float = 0.1,
    ) -> None:
        """Store fake risk scores."""
        self._assessment = GuardianAssessment(
            hallucination_risk=hallucination_risk,
            factual_consistency_risk=factual_consistency_risk,
            unsafe_content_risk=unsafe_content_risk,
            terminology_risk=terminology_risk,
            raw_response="fake guardian response",
        )

    def assess(
        self,
        *,
        fused_terms: list[FusedMedicalTerm],
        simplified_report: str,
    ) -> GuardianAssessment:
        """Return deterministic risk scores."""
        return self._assessment


def _fused_term() -> FusedMedicalTerm:
    """Build a fused term for validation tests."""
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


def _request_payload(report: str) -> dict[str, object]:
    """Build validation request JSON."""
    return {
        "simplified_report": report,
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


def test_granite_guardian_validation_approves_grounded_output() -> None:
    """Grounded text with low Guardian risk should be approved."""
    service = GraniteGuardianValidationService(
        Settings(),
        guardian_backend=FakeGuardianBackend(),
    )

    result = service.validate(
        fused_terms=[_fused_term()],
        simplified_report=(
            "Simplified Report: A patient has hypertension, a chronic condition "
            "where blood pressure remains higher than normal. Key Terms: "
            "Hypertension means high blood pressure."
        ),
    )

    assert result.validation_passed is True
    assert result.action is ValidationAction.APPROVE
    assert all(check.passed for check in result.checks)


def test_granite_guardian_validation_recommends_regeneration_for_missing_term() -> None:
    """Missing required terminology should trigger regeneration."""
    service = GraniteGuardianValidationService(
        Settings(),
        guardian_backend=FakeGuardianBackend(),
    )

    result = service.validate(
        fused_terms=[_fused_term()],
        simplified_report="Simplified Report: This is high blood pressure.",
    )

    assert result.validation_passed is False
    assert result.action is ValidationAction.REGENERATE
    assert any(check.name == "required_terms_preserved" for check in result.checks)


def test_granite_guardian_validation_rejects_unsafe_advice() -> None:
    """Unsupported care advice should be rejected."""
    service = GraniteGuardianValidationService(
        Settings(),
        guardian_backend=FakeGuardianBackend(),
    )

    result = service.validate(
        fused_terms=[_fused_term()],
        simplified_report=(
            "Hypertension means high blood pressure. It needs emergency treatment."
        ),
    )

    assert result.validation_passed is False
    assert result.action is ValidationAction.REJECT
    assert any(check.name == "unsafe_or_unsupported_advice" for check in result.checks)


def test_granite_guardian_validation_requires_inputs() -> None:
    """Validation requires fused terms and a generated report."""
    service = GraniteGuardianValidationService(
        Settings(),
        guardian_backend=FakeGuardianBackend(),
    )

    try:
        service.validate(fused_terms=[], simplified_report="text")
    except ValidationError as exc:
        assert "At least one" in str(exc)
    else:
        raise AssertionError("Expected ValidationError")


def test_granite_guardian_endpoint_returns_validation_json() -> None:
    """The API should expose Stage 9 validation results."""
    app = create_app()
    client = TestClient(app)

    from app.api.routes.simplify import get_granite_guardian_validator

    app.dependency_overrides[get_granite_guardian_validator] = lambda: GraniteGuardianValidationService(
        Settings(),
        guardian_backend=FakeGuardianBackend(),
    )

    response = client.post(
        "/api/v1/reports/validate",
        json=_request_payload(
            "Hypertension is a chronic condition where blood pressure remains higher "
            "than normal. Hypertension means high blood pressure."
        ),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["validation_passed"] is True
    assert payload["action"] == "approve"
    assert payload["guardian_assessment"]["raw_response"] == "fake guardian response"

    app.dependency_overrides.clear()

"""Tests for Stage 11 final API."""

from fastapi.testclient import TestClient

from app.evaluation.report_evaluation import (
    BERTScoreMetrics,
    EvaluationResult,
    MedicalConsistencyMetrics,
    ReadabilityMetrics,
)
from app.fusion.medical_fusion import FusedMedicalTerm
from app.main import create_app
from app.pipelines.medical_report_pipeline import (
    HighlightedDifficultTerm,
    PipelineResult,
)
from app.services.granite_guardian import (
    GuardianAssessment,
    ValidationAction,
    ValidationCheck,
    ValidationResult,
)


class FakeFinalPipeline:
    """Deterministic full pipeline test double."""

    def simplify_text(self, text: str) -> PipelineResult:
        """Return a fixed final result for text input."""
        assert "Hypertension" in text
        return _pipeline_result()

    def simplify_file(
        self,
        *,
        filename: str,
        content: bytes,
        content_type: str | None,
    ) -> PipelineResult:
        """Return a fixed final result for file input."""
        assert filename == "report.txt"
        assert content
        return _pipeline_result()


def _pipeline_result() -> PipelineResult:
    """Build a representative final pipeline result."""
    fused_term = FusedMedicalTerm(
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
    return PipelineResult(
        simplified_report=(
            "Simplified Report: A patient has hypertension, a chronic condition "
            "where blood pressure remains higher than normal. Key Terms: "
            "Hypertension means high blood pressure."
        ),
        highlighted_difficult_terms=[
            HighlightedDifficultTerm(
                term="Hypertension",
                difficulty=0.826,
                confidence=0.9393,
                entity_type="disease",
                section_type="diagnosis",
                meaning="High blood pressure.",
            )
        ],
        explanations=[
            {
                "term": "Hypertension",
                "explanation": "High blood pressure.",
                "difficulty": 0.826,
                "confidence": 0.9393,
            }
        ],
        confidence=0.9393,
        evaluation=EvaluationResult(
            bertscore=BERTScoreMetrics(precision=0.91, recall=0.92, f1=0.915),
            semantic_similarity=0.88,
            readability=ReadabilityMetrics(
                flesch_kincaid_grade_level=7.6,
                flesch_reading_ease=49.48,
            ),
            medical_consistency=MedicalConsistencyMetrics(
                score=1.0,
                terms_preserved=True,
                meanings_preserved=True,
                unsupported_numbers=[],
                missing_terms=[],
                missing_meanings=[],
            ),
            warnings=[],
        ),
        validation=ValidationResult(
            validation_passed=True,
            action=ValidationAction.APPROVE,
            checks=[
                ValidationCheck(
                    name="hallucination_detection",
                    passed=True,
                    score=0.9,
                    details="Granite Guardian hallucination risk.",
                )
            ],
            guardian_assessment=GuardianAssessment(
                hallucination_risk=0.1,
                factual_consistency_risk=0.1,
                unsafe_content_risk=0.1,
                terminology_risk=0.1,
                raw_response="{}",
            ),
            warnings=[],
        ),
        fused_terms=[fused_term],
        warnings=[],
    )


def test_final_simplify_endpoint_returns_complete_json_contract() -> None:
    """The final JSON endpoint should return report, terms, validation, and metrics."""
    app = create_app()
    client = TestClient(app)

    from app.api.routes.simplify import get_medical_report_pipeline

    app.dependency_overrides[get_medical_report_pipeline] = lambda: FakeFinalPipeline()

    response = client.post(
        "/api/v1/reports/simplify",
        json={"text": "Diagnosis: Hypertension"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Hypertension" in payload["simplified_report"]
    assert payload["highlighted_difficult_terms"][0]["term"] == "Hypertension"
    assert payload["explanations"][0]["explanation"] == "High blood pressure."
    assert payload["confidence"] == 0.9393
    assert payload["evaluation_scores"]["bertscore"]["f1"] == 0.915
    assert payload["validation"]["action"] == "approve"

    app.dependency_overrides.clear()


def test_final_file_endpoint_returns_json_response() -> None:
    """The final file endpoint should support uploaded report files."""
    app = create_app()
    client = TestClient(app)

    from app.api.routes.simplify import get_medical_report_pipeline

    app.dependency_overrides[get_medical_report_pipeline] = lambda: FakeFinalPipeline()

    response = client.post(
        "/api/v1/reports/simplify/file",
        files={"file": ("report.txt", b"Diagnosis: Hypertension", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json()["highlighted_difficult_terms"][0]["meaning"] == (
        "High blood pressure."
    )

    app.dependency_overrides.clear()

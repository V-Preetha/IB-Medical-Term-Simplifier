"""Tests for Stage 8 Qwen3 simplification."""

from fastapi.testclient import TestClient

from app.fusion.medical_fusion import FusedMedicalTerm
from app.main import create_app
from app.config.settings import Settings
from app.services.qwen_simplification import (
    QwenSimplificationService,
    SimplificationError,
    _clean_generation,
    _ground_patient_output,
    _strip_reasoning,
)


class FakeQwenBackend:
    """Deterministic Qwen3 generation backend for tests."""

    model_name = "fake-qwen3-instruct"

    def generate(self, prompt: str) -> str:
        """Return a fixed simplification and assert prompt safety rules exist."""
        assert "Structured fused medical representation" in prompt
        assert "Never change medication names" in prompt
        assert "Hypertension" in prompt
        return (
            "Simplified Report:\n"
            "Hypertension means high blood pressure. It is a long-term condition.\n\n"
            "Key Terms:\n"
            "- Hypertension: High blood pressure."
        )


class StaticPromptLoader:
    """Prompt loader test double."""

    def load(self) -> str:
        """Return a minimal prompt template with required rules."""
        return (
            "Never change medication names.\n"
            "Structured fused medical representation:\n"
            "{fused_terms_json}"
        )


def _fused_term() -> FusedMedicalTerm:
    """Build a fused medical term for tests."""
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
        ambiguity_resolution="Resolved as the chronic disease.",
        modernbert_embedding=[0.6, 0.8],
        semantic_embedding=[1.0, 0.0],
        matched_concept="hypertension",
    )


def test_qwen_simplification_uses_fused_terms_only() -> None:
    """Qwen3 simplification should produce report and term explanations."""
    service = QwenSimplificationService(
        Settings(),
        generation_backend=FakeQwenBackend(),
        prompt_loader=StaticPromptLoader(),
    )

    result = service.simplify([_fused_term()])

    assert result.model_name == "fake-qwen3-instruct"
    assert "Simplified Report:" in result.simplified_report
    assert result.term_explanations[0].term == "Hypertension"
    assert result.term_explanations[0].explanation == "High blood pressure."


def test_qwen_simplification_requires_fused_terms() -> None:
    """Stage 8 requires Stage 7 fused structured input."""
    service = QwenSimplificationService(
        Settings(),
        generation_backend=FakeQwenBackend(),
        prompt_loader=StaticPromptLoader(),
    )

    try:
        service.simplify([])
    except SimplificationError as exc:
        assert "At least one" in str(exc)
    else:
        raise AssertionError("Expected SimplificationError")


def test_qwen_simplification_strips_qwen_thinking_tags() -> None:
    """Patient-facing output should not include Qwen3 reasoning traces."""
    generated = (
        "<think>I should not show this.</think>\n"
        "Simplified Report:\nHypertension means high blood pressure."
    )

    assert _strip_reasoning(generated) == (
        "Simplified Report:\nHypertension means high blood pressure."
    )


def test_qwen_simplification_removes_echoed_structured_input() -> None:
    """Patient output should not include echoed structured JSON."""
    generated = (
        "Simplified Report:\nHypertension means high blood pressure.\n\n"
        "Structured fused medical representation:\n[{\"term\":\"Hypertension\"}]"
    )

    assert _clean_generation(generated) == (
        "Simplified Report:\nHypertension means high blood pressure."
    )


def test_qwen_simplification_removes_unsupported_advice_and_metadata() -> None:
    """Generated output should stay grounded in fused facts."""
    generated = (
        "Simplified Report:\n"
        "Hypertension means high blood pressure. "
        "It can be managed with medication and lifestyle changes.\n\n"
        "Key Terms:\n"
        "- Hypertension: High blood pressure.\n"
        "- Confidence: 0.9393\n"
        "- Ambiguity Resolution: Resolved as chronic disease."
    )

    assert _ground_patient_output(text=generated, fused_terms=[_fused_term()]) == (
        "Simplified Report:\nHypertension means high blood pressure.\n\n"
        "Key Terms:\n- Hypertension: High blood pressure."
    )


def test_qwen_simplification_endpoint_uses_dependency_injected_service() -> None:
    """The API should expose Qwen3 generated simplification."""
    app = create_app()
    client = TestClient(app)

    from app.api.routes.simplify import get_qwen_simplification_service

    app.dependency_overrides[get_qwen_simplification_service] = lambda: QwenSimplificationService(
        Settings(),
        generation_backend=FakeQwenBackend(),
        prompt_loader=StaticPromptLoader(),
    )

    response = client.post(
        "/api/v1/reports/simplify/from-fusion",
        json={
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
                    "ambiguity_resolution": "Resolved as the chronic disease.",
                    "modernbert_embedding": [0.6, 0.8],
                    "semantic_embedding": [1.0, 0.0],
                    "matched_concept": "hypertension",
                }
            ]
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "fake-qwen3-instruct"
    assert payload["term_explanations"][0]["term"] == "Hypertension"
    assert "Hypertension means high blood pressure" in payload["simplified_report"]

    app.dependency_overrides.clear()

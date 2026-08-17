"""MVP simplification and translation API contract tests."""

from fastapi.testclient import TestClient

from app.main import create_app
from app.ner.service import MedicalNERService
from app.ocr.providers.runtime import ProviderContainer
from app.simplification.contracts import (
    BaseSimplificationProvider,
    MedicalTermExplanation,
    ProviderSimplificationResult,
    SimplificationLevelResult,
    SimplificationProviderMetadata,
)
from app.simplification.service import SimplificationService
from app.translation.contracts import BaseTranslationProvider, TranslationProviderMetadata
from app.translation.service import TranslationService
from tests.ner.fakes import FakeNERProvider
from tests.ocr.fakes import FakeOCRProvider, FakePostProcessor


class FakeSimplificationProvider(BaseSimplificationProvider):
    async def initialize(self, *, strict: bool = True) -> None:
        del strict

    def simplify(self, text, entities, linked_concepts=()):
        del text, entities, linked_concepts
        levels = tuple(
            SimplificationLevelResult(
                level=level,
                simplified_report="The report records blood sugar of 126 mg/dL.",
                medical_terms_explained=(
                    MedicalTermExplanation("Glucose", "blood sugar"),
                ),
                important_findings=("Glucose: 126 mg/dL",),
                suggested_questions_for_doctor=(
                    "What does this documented glucose result mean?",
                ),
            )
            for level in ("clinical", "general_public", "child_friendly")
        )
        return ProviderSimplificationResult(levels, 20, 30, 4.5)

    def metadata(self):
        return SimplificationProviderMetadata(
            "qwen3",
            "Qwen/Qwen3-0.6B",
            "test-revision",
            "test-prompt-v1",
            "cpu",
            True,
            "ready",
            {"local_files_only": True},
        )

    async def shutdown(self) -> None:
        return None


class FakeTranslationProvider(BaseTranslationProvider):
    async def initialize(self, *, strict: bool = True) -> None:
        del strict

    def translate(self, text, source_language, target_language):
        del text, source_language, target_language
        return "??????? ??? ???? ?????? 126 mg/dL ???? ???"

    def metadata(self):
        return TranslationProviderMetadata(
            "indictrans2",
            "ai4bharat/indictrans2-en-indic-dist-200M",
            "test-revision",
            "cpu",
            True,
            "ready",
            {"local_files_only": True},
        )

    async def shutdown(self) -> None:
        return None


def _application():
    return create_app(
        provider_container=ProviderContainer(FakeOCRProvider(), FakePostProcessor()),
        ner_service=MedicalNERService(FakeNERProvider("biomedical-ner-all")),
        simplification_service=SimplificationService(FakeSimplificationProvider()),
        translation_service=TranslationService(FakeTranslationProvider()),
    )


def test_mvp_stage_apis_and_openapi() -> None:
    with TestClient(_application()) as client:
        simplification = client.post(
            "/api/v1/simplifications",
            json={
                "text": "Glucose was 126 mg/dL.",
                "entities": [
                    {
                        "text": "Glucose",
                        "label": "Laboratory Test",
                        "start": 0,
                        "end": 7,
                        "confidence": 0.98,
                    }
                ],
            },
        )
        translation = client.post(
            "/api/v1/translations",
            json={
                "text": simplification.json()["simplified_report"],
                "source_language": "eng_Latn",
                "target_language": "hin_Deva",
            },
        )
        schema = client.get("/openapi.json").json()

    assert simplification.status_code == 200
    assert simplification.json()["review_required"] is True
    assert simplification.json()["model_version"] == "test-revision"
    assert translation.status_code == 200
    assert "126 mg/dL" in translation.json()["translated_text"]
    assert translation.json()["review_required"] is True
    assert "/api/v1/simplifications" in schema["paths"]
    assert "/api/v1/translations" in schema["paths"]


def test_demo_marks_non_functional_stages_without_fabricating_output() -> None:
    """Entity Linking, Relation Extraction, and TTS remain visibly non-functional.

    The Pipeline Test Console (see test_engineering_demo.py) labels these
    Runtime Pending / Frozen rather than the older "Deferred for MVP" wording,
    and drives Simplification/Verification/Translation through the current
    production endpoints instead of the deprecated compatibility routes.
    """
    with TestClient(_application()) as client:
        page = client.get("/engineering-demo").text
        script = client.get("/static/engineering_demo.js").text

    assert page.count("Runtime Pending") >= 2
    assert "Frozen" in page
    assert "Translated report will appear here." in page
    assert "/api/v1/simplify" in script
    assert "/api/v1/verification" in script
    assert "/api/v1/translations" in script
    assert "/api/v1/simplifications" not in script

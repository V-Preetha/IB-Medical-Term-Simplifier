"""Verify stage outputs are passed through the MVP patient workflow."""

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


class _SimplificationProvider(BaseSimplificationProvider):
    async def initialize(self, *, strict: bool = True) -> None:
        del strict

    def simplify(self, text, entities, linked_concepts=()):
        del entities, linked_concepts
        assert text == "blood pressure (BP) 120 / 80 hemoglobin 6.5%"
        levels = tuple(
            SimplificationLevelResult(
                level=level,
                simplified_report="The report records hemoglobin of 6.5%.",
                medical_terms_explained=(
                    MedicalTermExplanation("hemoglobin", "a blood component"),
                ),
                important_findings=("hemoglobin 6.5%",),
                suggested_questions_for_doctor=("What does this report record?",),
            )
            for level in ("clinical", "general_public", "child_friendly")
        )
        return ProviderSimplificationResult(levels, 10, 12, 2.0)

    def metadata(self):
        return SimplificationProviderMetadata(
            "qwen3",
            "Qwen/Qwen3-0.6B",
            "test-revision",
            "test-prompt-v2",
            "cpu",
            True,
            "ready",
            {"local_files_only": True},
        )

    async def shutdown(self) -> None:
        return None


class _TranslationProvider(BaseTranslationProvider):
    def __init__(self) -> None:
        self.received: tuple[str, ...] = ()

    async def initialize(self, *, strict: bool = True) -> None:
        del strict

    def translate(self, text, source_language, target_language):
        return self.translate_batch((text,), source_language, target_language)[0]

    def translate_batch(self, texts, source_language, target_language):
        assert source_language == "eng_Latn"
        assert target_language == "hin_Deva"
        self.received = texts
        return tuple(f"Hindi: {text}" for text in texts)

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


def test_http_workflow_passes_ocr_to_ner_to_simplification_to_translation() -> None:
    translation_provider = _TranslationProvider()
    application = create_app(
        provider_container=ProviderContainer(FakeOCRProvider(), FakePostProcessor()),
        ner_service=MedicalNERService(FakeNERProvider("biomedical-ner-all")),
        simplification_service=SimplificationService(_SimplificationProvider()),
        translation_service=TranslationService(translation_provider),
    )
    with TestClient(application) as client:
        ocr = client.post(
            "/api/v1/ocr",
            files={"file": ("synthetic.png", b"synthetic-content", "image/png")},
        )
        ner = client.post("/api/v1/ner", json={"text": ocr.json()["normalized_text"]})
        simplified = client.post(
            "/api/v1/simplify",
            json={"text": ocr.json()["normalized_text"], "entities": ner.json()["entities"]},
        )
        translated = client.post(
            "/api/v1/translations",
            json={
                "text": simplified.json()["clinical"]["simplified_report"],
                "source_language": "eng_Latn",
                "target_language": "hin_Deva",
            },
        )

    assert ocr.status_code == 201
    assert ner.status_code == 200
    assert simplified.status_code == 200
    assert translated.status_code == 200
    assert translation_provider.received == (
        simplified.json()["clinical"]["simplified_report"],
    )
    assert translated.json()["translated_text"].endswith("hemoglobin of 6.5%.")

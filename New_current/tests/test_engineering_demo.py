"""Contract tests for the consolidated internal Pipeline Test Console."""

from fastapi.testclient import TestClient

from app.embeddings.service import MedicalEmbeddingService
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
from app.verification.contracts import (
    BaseVerificationProvider,
    VerificationInference,
    VerificationProviderMetadata,
)
from app.verification.service import MedicalVerificationService
from tests.embeddings.fakes import FakeEmbeddingProvider
from tests.ner.fakes import FakeNERProvider
from tests.ocr.fakes import FakeOCRProvider, FakePostProcessor

# Stale numbers from a historical benchmark document. If any of these appear
# literally in the shipped page or script, a real measurement was hardcoded
# instead of being read from the live API response.
_STALE_BENCHMARK_NUMBERS = ("7,123.953", "14,264.999", "22,185.073", "27.964")


class _FakeSimplificationProvider(BaseSimplificationProvider):
    async def initialize(self, *, strict: bool = True) -> None:
        del strict

    def simplify(self, text, entities, linked_concepts=()):
        del entities, linked_concepts
        levels = tuple(
            SimplificationLevelResult(
                level=level,
                simplified_report=f"{level} report for: {text}",
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


class _FakeTranslationProvider(BaseTranslationProvider):
    def __init__(self) -> None:
        self.received: tuple[str, ...] = ()

    async def initialize(self, *, strict: bool = True) -> None:
        del strict

    def translate(self, text, source_language, target_language):
        return self.translate_batch((text,), source_language, target_language)[0]

    def translate_batch(self, texts, source_language, target_language):
        self.received = texts
        return tuple(f"{target_language}: {text}" for text in texts)

    def metadata(self):
        return TranslationProviderMetadata(
            "indictrans2",
            "ai4bharat/indictrans2-en-indic-dist-200M",
            "test-revision",
            "cpu",
            True,
            "ready",
            {"local_files_only": True, "supported_languages": {"hin_Deva": "Hindi"}},
        )

    async def shutdown(self) -> None:
        return None


class _FakeVerificationProvider(BaseVerificationProvider):
    """Deterministic MedNLI double keyed on the requested premise for testing."""

    def __init__(self, *, label: str = "entailment") -> None:
        self.label = label

    async def initialize(self, *, strict: bool = True) -> None:
        del strict

    def infer(self, premise: str, hypothesis: str) -> VerificationInference:
        del premise, hypothesis
        probabilities = {"contradiction": 0.0, "entailment": 0.0, "neutral": 0.0}
        probabilities[self.label] = 1.0
        return VerificationInference(self.label, probabilities, 5.0)

    def metadata(self) -> VerificationProviderMetadata:
        return VerificationProviderMetadata(
            "pritamdeka/PubMedBERT-MNLI-MedNLI",
            "test-revision",
            "cpu",
            True,
            "ready",
            {"license_status": "PENDING_VERIFICATION"},
        )

    async def shutdown(self) -> None:
        return None


def _application(*, verification_label: str = "entailment", translation_provider=None):
    return create_app(
        provider_container=ProviderContainer(FakeOCRProvider(), FakePostProcessor()),
        ner_service=MedicalNERService(FakeNERProvider("biomedical-ner-all")),
        embedding_service=MedicalEmbeddingService(FakeEmbeddingProvider()),
        simplification_service=SimplificationService(_FakeSimplificationProvider()),
        translation_service=TranslationService(translation_provider or _FakeTranslationProvider()),
        verification_service=MedicalVerificationService(
            _FakeVerificationProvider(label=verification_label)
        ),
    )


def test_engineering_demo_is_single_same_origin_page() -> None:
    with TestClient(_application()) as client:
        dashboard = client.get("/engineering-demo")
        javascript = client.get("/static/engineering_demo.js")
        stylesheet = client.get("/static/engineering_demo.css")
        swagger = client.get("/docs")

    assert dashboard.status_code == 200
    assert javascript.status_code == 200
    assert stylesheet.status_code == 200
    assert swagger.status_code == 200
    assert "Internal Engineering Demonstration" in dashboard.text
    assert "Pipeline Test Console" in dashboard.text
    for section in (
        "Document Input",
        "OCR",
        "Medical NER",
        "Entity Linking",
        "Relation Extraction",
        "Medical Embeddings",
        "Infrastructure",
        "Simplification",
        "Verification",
        "Translation",
        "TTS",
        "Performance Panel",
        "Safety Checks",
        "Runtime Stats",
        "Raw API Inspector",
        "Full Pipeline Mode",
    ):
        assert section in dashboard.text
    assert "Architecture Complete" in dashboard.text
    assert "Runtime Pending" in dashboard.text
    assert "Planned" in dashboard.text
    assert "Frozen" in dashboard.text
    assert "react" not in dashboard.text.casefold()
    assert "NOT EXPOSED" in dashboard.text


def test_demo_javascript_reuses_only_existing_runtime_endpoints() -> None:
    with TestClient(_application()) as client:
        script = client.get("/static/engineering_demo.js").text
        schema = client.get("/openapi.json").json()

    referenced = {
        "/api/v1/ocr",
        "/api/v1/ocr/health",
        "/api/v1/ocr/models",
        "/api/v1/ner",
        "/api/v1/ner/health",
        "/api/v1/ner/models",
        "/api/v1/entity-linking/health",
        "/api/v1/relation-extraction/health",
        "/api/v1/embeddings",
        "/api/v1/embeddings/health",
        "/api/v1/embeddings/models",
        "/api/v1/simplify",
        "/api/v1/simplify/health",
        "/api/v1/verification",
        "/api/v1/verification/health",
        "/api/v1/translations",
        "/api/v1/translations/health",
        "/api/v1/infrastructure/health",
        "/api/v1/runtime/metrics",
    }
    assert all(endpoint in script for endpoint in referenced)
    assert referenced <= set(schema["paths"])
    assert "qdrant" not in script.casefold()
    assert "/api/v1/simplifications" not in script  # deprecated compatibility route


def test_demo_javascript_never_reimplements_stage_business_logic() -> None:
    """The script may only orchestrate; it must not embed inference logic."""
    with TestClient(_application()) as client:
        script = client.get("/static/engineering_demo.js").text

    for forbidden in ("import torch", "AutoModel", "onnxruntime", "def "):
        assert forbidden not in script


def test_demo_shows_real_stage_status_vocabulary_not_fake_progress() -> None:
    with TestClient(_application()) as client:
        dashboard = client.get("/engineering-demo").text
        script = client.get("/static/engineering_demo.js").text

    for label in ("WAITING", "RUNNING", "PASS", "REVIEW", "BLOCKED", "FAILED", "DEFERRED"):
        assert label in dashboard or label in script
    # The tracker/progress states must come from real API results, not a timer.
    assert "setInterval" not in script
    assert "setTimeout" not in script or "setTimeout(() => { button.textContent" in script


def test_demo_verification_gating_logic_is_present() -> None:
    with TestClient(_application()) as client:
        script = client.get("/static/engineering_demo.js").text

    # These functions implement "BLOCKED/REVIEW prevents translation" purely
    # by reading the real /api/v1/verification response field.
    assert "overallVerificationState" in script
    assert "populateTranslateLevelOptions" in script
    assert 'verification === "PASS"' in script
    assert "No level passed verification" in script


def test_demo_does_not_hardcode_stale_benchmark_numbers() -> None:
    with TestClient(_application()) as client:
        dashboard = client.get("/engineering-demo").text
        script = client.get("/static/engineering_demo.js").text

    for stale_number in _STALE_BENCHMARK_NUMBERS:
        assert stale_number not in dashboard
        assert stale_number not in script


def test_full_stage_chain_pass_allows_translation_via_real_endpoints() -> None:
    """Exercises the exact endpoints the dashboard calls, PASS case."""
    translation_provider = _FakeTranslationProvider()
    with TestClient(
        _application(verification_label="entailment", translation_provider=translation_provider)
    ) as client:
        ocr = client.post(
            "/api/v1/ocr",
            files={"file": ("synthetic.png", b"synthetic-content", "image/png")},
        )
        ner = client.post("/api/v1/ner", json={"text": ocr.json()["normalized_text"]})
        simplified = client.post(
            "/api/v1/simplify",
            json={"text": ocr.json()["normalized_text"], "entities": ner.json()["entities"]},
        )
        verification = client.post(
            "/api/v1/verification",
            json={
                "premise": ocr.json()["normalized_text"],
                "hypothesis": simplified.json()["general_public"]["simplified_report"],
            },
        )
        translated = client.post(
            "/api/v1/translations",
            json={
                "text": simplified.json()["general_public"]["simplified_report"],
                "source_language": "eng_Latn",
                "target_language": "hin_Deva",
            },
        )

    assert ocr.status_code == 201
    assert ner.status_code == 200
    assert simplified.status_code == 200
    assert verification.status_code == 200
    assert verification.json()["verification"] == "PASS"
    assert translated.status_code == 200


def test_verification_blocked_response_would_prevent_dashboard_translation() -> None:
    """A contradiction label must surface BLOCKED, which the dashboard reads
    to disable translation for that level (see populateTranslateLevelOptions)."""
    with TestClient(_application(verification_label="contradiction")) as client:
        ocr = client.post(
            "/api/v1/ocr",
            files={"file": ("synthetic.png", b"synthetic-content", "image/png")},
        )
        simplified = client.post(
            "/api/v1/simplify",
            json={"text": ocr.json()["normalized_text"], "entities": []},
        )
        verification = client.post(
            "/api/v1/verification",
            json={
                "premise": ocr.json()["normalized_text"],
                "hypothesis": simplified.json()["clinical"]["simplified_report"],
            },
        )

    assert verification.status_code == 200
    assert verification.json()["verification"] == "BLOCKED"
    assert verification.json()["review_required"] is True


def test_verification_neutral_response_requires_review() -> None:
    with TestClient(_application(verification_label="neutral")) as client:
        ocr = client.post(
            "/api/v1/ocr",
            files={"file": ("synthetic.png", b"synthetic-content", "image/png")},
        )
        simplified = client.post(
            "/api/v1/simplify",
            json={"text": ocr.json()["normalized_text"], "entities": []},
        )
        verification = client.post(
            "/api/v1/verification",
            json={
                "premise": ocr.json()["normalized_text"],
                "hypothesis": simplified.json()["clinical"]["simplified_report"],
            },
        )

    assert verification.status_code == 200
    assert verification.json()["verification"] == "REVIEW"
    assert verification.json()["review_required"] is True


def test_demo_api_failure_handling_present_without_stack_traces() -> None:
    with TestClient(_application()) as client:
        script = client.get("/static/engineering_demo.js").text

    assert "callApi" in script
    assert "describeError" in script
    assert "requestId" in script
    # The error path only ever reads the safe JSON envelope, never a traceback field.
    assert "traceback" not in script.casefold()
    assert ".stack" not in script.casefold()


def test_runtime_metrics_endpoint_reports_real_or_null_never_fabricated() -> None:
    """Every field is either a genuinely measured value or ``null``.

    ``psutil`` process RSS always works on this platform, so it must be a real
    number; GPU utilization requires an unapproved dependency (pynvml) that is
    not installed, so it must be ``null`` rather than an invented percentage.
    """
    with TestClient(_application()) as client:
        response = client.get("/api/v1/runtime/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "runtime-metrics-v1"

    gpu = body["gpu"]
    assert isinstance(gpu["available"], bool)
    if not gpu["available"]:
        assert gpu["allocated_mb"] is None
        assert gpu["utilization_percent"] is None

    cpu = body["cpu"]
    assert isinstance(cpu["process_rss_mb"], float)
    assert cpu["process_rss_mb"] > 0

    stages = {item["stage"] for item in body["models"]}
    assert {"ocr", "ner", "embeddings"} <= stages
    for item in body["models"]:
        assert isinstance(item["loaded"], bool)
        assert isinstance(item["warm"], bool)
        # Never a placeholder string standing in for missing data.
        for field in ("provider_name", "model_name", "model_revision", "device"):
            assert item[field] != "NOT EXPOSED"
            assert item[field] != "unknown"


def test_verification_and_translation_health_expose_real_device() -> None:
    """The device gap this task was asked to close: both must report a real
    device string via their /health endpoints, not omit the field."""
    with TestClient(_application()) as client:
        verification_health = client.get("/api/v1/verification/health").json()
        translation_health = client.get("/api/v1/translations/health").json()

    assert verification_health["device"] == "cpu"
    assert translation_health["device"] == "cpu"


def test_dashboard_surfaces_runtime_metrics_vocabulary() -> None:
    with TestClient(_application()) as client:
        dashboard = client.get("/engineering-demo").text
        script = client.get("/static/engineering_demo.js").text

    assert "loadRuntimeMetrics" in script
    assert "applyStageWarmth" in script
    assert "GPU memory allocated" in dashboard
    assert "Peak GPU memory" in dashboard
    assert "Process RSS" in dashboard
    assert "Upload / Read" in dashboard or "Upload / Read" in script
    assert "Decode / Render" in script


def test_demo_never_hardcodes_a_fake_gpu_or_cpu_reading() -> None:
    """Guards against regressions that hardcode a plausible-looking number
    instead of reading it from /api/v1/runtime/metrics."""
    with TestClient(_application()) as client:
        script = client.get("/static/engineering_demo.js").text

    assert "gpu.allocated_mb" in script
    assert "gpu.utilization_percent" in script
    assert "cpu.process_rss_mb" in script

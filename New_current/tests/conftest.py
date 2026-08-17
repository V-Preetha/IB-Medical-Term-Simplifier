"""Test-only runtime configuration."""

import os
from pathlib import Path

import pytest

_OCR_FIXTURE_DIRECTORY = Path(__file__).parent / "ocr" / "fixtures"

os.environ.setdefault("REPORT_CLINICAL_NER_ENABLED", "false")
os.environ.setdefault("REPORT_QWEN_SIMPLIFIER_ENABLED", "false")
os.environ.setdefault("OCR_PROVIDER", "qwen3-vl")
os.environ.setdefault("POSTPROCESSOR_PROVIDER", "symspell")
os.environ.setdefault("OCR_CONFIG__PROVIDER_VERSION", "contract-1.0.0")
os.environ.setdefault("OCR_CONFIG__MODEL_NAME", "qwen3-vl-test-contract")
os.environ.setdefault("OCR_CONFIG__MODEL_REVISION", "test-revision")
os.environ.setdefault("OCR_CONFIG__DEVICE", "cpu")
os.environ.setdefault("OCR_CONFIG__HF_CACHE_DIR", str(_OCR_FIXTURE_DIRECTORY))
os.environ.setdefault("OCR_CONFIG__LOCAL_FILES_ONLY", "true")
os.environ.setdefault("OCR_CONFIG__ALLOW_CPU_FALLBACK", "true")
os.environ.setdefault("OCR_CONFIG__DTYPE", "float32")
os.environ.setdefault("OCR_CONFIG__BATCH_SIZE", "2")
os.environ.setdefault("OCR_CONFIG__MAX_IMAGE_SIZE", "1024")
os.environ.setdefault("OCR_CONFIG__MAX_PAGES", "10")
os.environ.setdefault("OCR_CONFIG__PDF_RENDER_DPI", "96")
os.environ.setdefault("OCR_CONFIG__TIMEOUT_SECONDS", "30")
os.environ.setdefault("OCR_CONFIG__MAX_NEW_TOKENS", "128")
os.environ.setdefault("OCR_CONFIG__DO_SAMPLE", "false")
os.environ.setdefault("OCR_CONFIG__TEMPERATURE", "0.1")
os.environ.setdefault("OCR_CONFIG__TOP_P", "1.0")
os.environ.setdefault("OCR_CONFIG__NUM_BEAMS", "1")
os.environ.setdefault("OCR_CONFIG__PROMPT", "Transcribe the document exactly.")
os.environ.setdefault("OCR_CONFIG__PROMPT_VERSION", "test-prompt-v1")
os.environ.setdefault("OCR_CONFIG__CONFIDENCE_THRESHOLD", "0.60")
os.environ.setdefault(
    "OCR_CONFIG__CONFIDENCE_CALIBRATION_VERSION",
    "token-probability-uncalibrated-v1",
)
os.environ.setdefault("OCR_CONFIG__SEED", "0")
os.environ.setdefault("POSTPROCESSOR_CONFIG__PROVIDER_VERSION", "contract-1.0.0")
os.environ.setdefault(
    "POSTPROCESSOR_CONFIG__DICTIONARY_PATH",
    str(_OCR_FIXTURE_DIRECTORY / "symspell_frequency.txt"),
)
os.environ.setdefault("POSTPROCESSOR_CONFIG__DICTIONARY_VERSION", "test-version")
os.environ.setdefault(
    "POSTPROCESSOR_CONFIG__ABBREVIATION_DICTIONARY_PATH",
    str(_OCR_FIXTURE_DIRECTORY / "medical_abbreviations.json"),
)
os.environ.setdefault(
    "POSTPROCESSOR_CONFIG__ABBREVIATION_DICTIONARY_VERSION",
    "test-version",
)
os.environ.setdefault("POSTPROCESSOR_CONFIG__RULE_VERSION", "test-version")
os.environ.setdefault("POSTPROCESSOR_CONFIG__MAX_EDIT_DISTANCE", "2")
os.environ.setdefault("POSTPROCESSOR_CONFIG__PREFIX_LENGTH", "7")
os.environ.setdefault("POSTPROCESSOR_CONFIG__MIN_WORD_LENGTH", "4")
os.environ.setdefault("POSTPROCESSOR_CONFIG__MIN_FREQUENCY", "1")
os.environ.setdefault(
    "POSTPROCESSOR_CONFIG__PROTECTED_TERMS_PATH",
    str(_OCR_FIXTURE_DIRECTORY / "protected_medical_terms.json"),
)


class _LifecycleOnlyModel:
    def to(self, device: str):
        del device
        return self

    def eval(self):
        return self


@pytest.fixture(autouse=True)
def isolate_ocr_model_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace external checkpoint loading; provider behavior is tested separately."""

    from app.ocr.providers.implementations import Qwen3VLOCRProvider

    def load_runtime(provider) -> None:
        provider._resolve_torch_device()
        provider._processor = object()
        provider._model = _LifecycleOnlyModel()

    monkeypatch.setattr(Qwen3VLOCRProvider, "_load_runtime", load_runtime)

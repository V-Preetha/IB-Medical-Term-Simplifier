"""Provider integration tests with model libraries isolated at their loading boundary."""

import asyncio
import os
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import fitz
import pytest
import torch
from PIL import Image

from app.ocr.providers.config import ProviderSettings
from app.ocr.providers.contracts import ProviderDocument, ProviderHealthStatus, ProviderKind
from app.ocr.providers.errors import (
    ProviderConfigurationError,
    ProviderInitializationError,
    ProviderUnavailableError,
    UnsupportedDocumentError,
)
from app.ocr.providers.implementations import (
    MedicalPostProcessingResult,
    Qwen3VLOCRProvider,
    Qwen3VLOCRResult,
    SymSpellPostProcessor,
)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2


class _FakeQwenProcessor:
    tokenizer = _FakeTokenizer()

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return {"input_ids": torch.ones((len(messages), 2), dtype=torch.long)}

    def batch_decode(self, generated, **kwargs):
        del kwargs
        return [
            (f'{{"document_type":"scanned_pdf","text":"Synthetic page {int(row[0].item()) - 2}"}}')
            for row in generated
        ]


class _TranscriptionOnlyProcessor(_FakeQwenProcessor):
    def batch_decode(self, generated, **kwargs):
        del generated, kwargs
        return ["Synthetic transcription"]


class _FakeQwenModel:
    def generate(self, input_ids, **kwargs):
        del kwargs
        batch_size = input_ids.shape[0]
        device = input_ids.device
        first_tokens = torch.arange(3, 3 + batch_size, device=device).unsqueeze(1)
        end_tokens = torch.full((batch_size, 1), 2, device=device)
        generated = torch.cat((first_tokens, end_tokens), dim=1)
        first_scores = torch.zeros((batch_size, 3 + batch_size), device=device)
        end_scores = torch.zeros((batch_size, 3 + batch_size), device=device)
        for index in range(batch_size):
            first_scores[index, 3 + index] = 8
            end_scores[index, 2] = 8
        return SimpleNamespace(
            sequences=torch.cat((input_ids, generated), dim=1),
            scores=(first_scores, end_scores),
        )


class _OutOfMemoryQwenModel:
    def generate(self, **kwargs):
        del kwargs
        raise torch.OutOfMemoryError("synthetic out of memory")


def _provider_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "OCR_PROVIDER",
            "POSTPROCESSOR_PROVIDER",
        }
        or key.startswith(
            (
                "OCR_CONFIG__",
                "POSTPROCESSOR_CONFIG__",
            )
        )
    }


def _configuration(kind: ProviderKind, **overrides: str) -> dict[str, str]:
    configuration = dict(
        ProviderSettings.from_environment(_provider_environment()).configuration_for(kind)
    )
    configuration.update(overrides)
    return configuration


def _png_document() -> ProviderDocument:
    buffer = BytesIO()
    Image.new("RGB", (80, 40), "white").save(buffer, format="PNG")
    return ProviderDocument(content=buffer.getvalue(), file_type="png", filename="synthetic.png")


def _multipage_pdf_document(page_count: int = 2) -> ProviderDocument:
    pdf = fitz.open()
    try:
        for page_number in range(1, page_count + 1):
            page = pdf.new_page(width=200, height=100)
            page.insert_text((20, 50), f"Synthetic page {page_number}")
        content = pdf.tobytes()
    finally:
        pdf.close()
    return ProviderDocument(content=content, file_type="pdf", filename="synthetic.pdf")


def _digital_pdf_document() -> ProviderDocument:
    pdf = fitz.open()
    try:
        for page_number in range(1, 3):
            page = pdf.new_page(width=300, height=150)
            page.insert_text(
                (20, 50),
                (
                    f"Page {page_number}. Glucose was 126 mg/dL. "
                    "The patient was advised to review the documented result."
                ),
            )
        content = pdf.tobytes()
    finally:
        pdf.close()
    return ProviderDocument(content=content, file_type="pdf", filename="digital.pdf")


def _multipage_tiff_document() -> ProviderDocument:
    buffer = BytesIO()
    first = Image.new("RGB", (80, 40), "white")
    second = Image.new("RGB", (60, 30), "gray")
    first.save(buffer, format="TIFF", save_all=True, append_images=[second])
    return ProviderDocument(
        content=buffer.getvalue(),
        file_type="tiff",
        filename="synthetic.tiff",
    )


def _ready_ocr(**configuration: str) -> Qwen3VLOCRProvider:
    provider = Qwen3VLOCRProvider(_configuration(ProviderKind.OCR, **configuration))
    asyncio.run(provider.initialize())
    provider._processor = _FakeQwenProcessor()
    provider._model = _FakeQwenModel()
    return provider


def test_model_lifecycle_loads_once_and_releases(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def load(provider) -> None:
        nonlocal calls
        calls += 1
        provider._resolve_torch_device()
        provider._processor = _FakeQwenProcessor()
        provider._model = _FakeQwenModel()

    monkeypatch.setattr(Qwen3VLOCRProvider, "_load_runtime", load)
    provider = Qwen3VLOCRProvider(_configuration(ProviderKind.OCR))

    asyncio.run(provider.initialize())
    asyncio.run(provider.initialize())

    assert calls == 1
    assert provider.health().status is ProviderHealthStatus.READY
    assert provider.metadata().configuration["model_loading_time_ms"] >= 0

    asyncio.run(provider.shutdown())
    assert provider.health().status is ProviderHealthStatus.STOPPED
    assert provider._model is None


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU fallback requires absent CUDA")
def test_cuda_request_falls_back_to_cpu_when_configured() -> None:
    provider = _ready_ocr(device="cuda", allow_cpu_fallback="true")
    try:
        result = provider.process(_png_document())
        metadata = provider.metadata().configuration
    finally:
        asyncio.run(provider.shutdown())

    assert result.confidence is not None
    assert metadata["resolved_device"] == "cpu"
    assert metadata["cpu_fallback"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime is unavailable")
def test_qwen3_vl_runs_on_gpu() -> None:
    provider = _ready_ocr(device="cuda", dtype="float16")
    try:
        result = provider.process(_png_document())
        assert result.confidence is not None
        assert provider.metadata().configuration["resolved_device"].startswith("cuda")
    finally:
        asyncio.run(provider.shutdown())


def test_qwen3_vl_processes_single_page_image() -> None:
    provider = _ready_ocr(batch_size="1")
    try:
        result = provider.process(_png_document())
    finally:
        asyncio.run(provider.shutdown())

    assert isinstance(result, Qwen3VLOCRResult)
    assert result.text == "Synthetic page 1"
    assert result.document_type == "scanned_pdf"
    assert result.confidence is not None and result.confidence > 0.9
    assert len(result.pages) == 1
    assert result.pages[0].page_number == 1
    assert result.statistics is not None
    assert result.statistics.page_count == 1
    assert result.statistics.resolved_device == "cpu"
    assert result.provider_metadata is not None


def test_qwen3_vl_transcription_only_mode_returns_raw_text_and_unknown_type() -> None:
    provider = _ready_ocr(transcription_only="true")
    provider._processor = _TranscriptionOnlyProcessor()
    try:
        result = provider.process(_png_document())
    finally:
        asyncio.run(provider.shutdown())

    assert result.text == "Synthetic transcription"
    assert result.document_type == "unknown"
    assert result.confidence is not None


def test_qwen3_vl_rejects_invalid_transcription_only_configuration() -> None:
    provider = Qwen3VLOCRProvider(_configuration(ProviderKind.OCR, transcription_only="maybe"))

    with pytest.raises(ProviderConfigurationError, match="transcription_only"):
        asyncio.run(provider.initialize())


def test_qwen3_vl_uses_direct_local_checkpoint_with_revision_provenance(
    tmp_path: Path,
) -> None:
    revision = "a" * 40
    checkpoint = tmp_path / "qwen"
    provenance = checkpoint / ".cache" / "huggingface" / "trees"
    provenance.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (provenance / f"{revision}.json").write_text("{}", encoding="utf-8")
    provider = Qwen3VLOCRProvider(
        _configuration(
            ProviderKind.OCR,
            hf_cache_dir=str(checkpoint),
            model_revision=revision,
            local_files_only="false",
        )
    )

    source, loading = provider._pretrained_source_and_configuration()

    assert source == str(checkpoint.resolve())
    assert loading == {"local_files_only": True}


def test_qwen3_vl_rejects_local_checkpoint_without_revision_provenance(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "qwen"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    provider = Qwen3VLOCRProvider(
        _configuration(
            ProviderKind.OCR,
            hf_cache_dir=str(checkpoint),
            model_revision="a" * 40,
        )
    )

    with pytest.raises(ProviderConfigurationError, match="provenance"):
        provider._pretrained_source_and_configuration()


def test_qwen3_vl_processes_multi_page_pdf_in_order() -> None:
    provider = _ready_ocr(batch_size="2")
    try:
        result = provider.process(_multipage_pdf_document())
    finally:
        asyncio.run(provider.shutdown())

    assert result.text == "Synthetic page 1\n\nSynthetic page 2"
    assert [page.page_number for page in result.pages] == [1, 2]
    assert result.statistics is not None
    assert result.statistics.page_count == 2
    assert result.statistics.batch_count == 1
    assert result.statistics.pages_per_second > 0


def test_digital_pdf_uses_native_text_layer_without_model_generation() -> None:
    provider = _ready_ocr()
    provider._model = _OutOfMemoryQwenModel()
    try:
        result = provider.process(_digital_pdf_document())
    finally:
        asyncio.run(provider.shutdown())

    assert result.document_type == "digital_pdf"
    assert "Glucose was 126 mg/dL" in result.text
    assert result.confidence is None
    assert result.confidence_method is None
    assert result.statistics is not None
    assert result.statistics.page_count == 2
    assert result.statistics.batch_count == 0


def test_qwen3_vl_processes_multi_page_tiff_in_order() -> None:
    provider = _ready_ocr(batch_size="2")
    try:
        result = provider.process(_multipage_tiff_document())
    finally:
        asyncio.run(provider.shutdown())

    assert result.text == "Synthetic page 1\n\nSynthetic page 2"
    assert [page.page_number for page in result.pages] == [1, 2]


def test_post_processing_runs_regex_dictionary_and_symspell() -> None:
    provider = SymSpellPostProcessor(_configuration(ProviderKind.POSTPROCESSOR))
    asyncio.run(provider.initialize())
    try:
        result = provider.normalize(
            "BP 120 / 80  haemoglobn 6 . 5 % Metformin",
            document_type="printed_image",
        )
    finally:
        asyncio.run(provider.shutdown())

    assert isinstance(result, MedicalPostProcessingResult)
    assert "blood pressure (BP)" in result.normalized_text
    assert "hemoglobin" in result.normalized_text
    assert "6.5%" in result.normalized_text
    assert "Metformin" in result.normalized_text
    assert [stage.stage for stage in result.stages] == [
        "regex",
        "medical_abbreviation_dictionary",
        "symspell",
    ]
    assert result.total_corrections >= 3
    assert all(stage.latency_ms >= 0 for stage in result.stages)

    asyncio.run(provider.initialize())
    try:
        repeated = provider.normalize(result.normalized_text, document_type="printed_image")
    finally:
        asyncio.run(provider.shutdown())
    assert repeated.normalized_text == result.normalized_text


def test_invalid_model_configuration_fails_before_loading() -> None:
    provider = Qwen3VLOCRProvider(_configuration(ProviderKind.OCR, batch_size="0"))

    with pytest.raises(ProviderConfigurationError, match="batch_size"):
        asyncio.run(provider.initialize())


def test_missing_or_corrupted_checkpoint_fails_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_loading(provider) -> None:
        del provider
        raise ProviderInitializationError("Synthetic checkpoint is corrupt.")

    monkeypatch.setattr(Qwen3VLOCRProvider, "_load_runtime", fail_loading)
    provider = Qwen3VLOCRProvider(_configuration(ProviderKind.OCR))

    with pytest.raises(ProviderInitializationError, match="corrupt"):
        asyncio.run(provider.initialize())
    assert provider.health().status is ProviderHealthStatus.UNAVAILABLE


def test_invalid_pdf_and_image_are_rejected() -> None:
    provider = _ready_ocr()
    try:
        with pytest.raises(UnsupportedDocumentError, match="PDF signature"):
            provider.process(ProviderDocument(content=b"not-a-pdf", file_type="pdf"))
        with pytest.raises(UnsupportedDocumentError, match="cannot be decoded"):
            provider.process(ProviderDocument(content=b"not-an-image", file_type="png"))
    finally:
        asyncio.run(provider.shutdown())


def test_out_of_memory_is_reported_as_provider_unavailable() -> None:
    provider = _ready_ocr()
    provider._model = _OutOfMemoryQwenModel()
    try:
        with pytest.raises(ProviderUnavailableError, match="exhausted"):
            provider.process(_png_document())
    finally:
        asyncio.run(provider.shutdown())


def test_inference_timeout_is_reported_as_provider_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter((0.0, 0.0, 1.0))
    monkeypatch.setattr(
        "app.ocr.providers.implementations.monotonic",
        lambda: next(clock),
    )
    provider = _ready_ocr(timeout_seconds="0.001")
    try:
        with pytest.raises(ProviderUnavailableError, match="timeout"):
            provider.process(_png_document())
    finally:
        asyncio.run(provider.shutdown())

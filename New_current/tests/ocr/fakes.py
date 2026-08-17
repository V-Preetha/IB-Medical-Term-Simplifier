"""Synthetic provider implementations for application and API boundary tests."""

from datetime import UTC, datetime
from typing import Any

from app.ocr.providers.contracts import (
    BaseOCRProvider,
    BasePostProcessor,
    OCRProviderResult,
    PostProcessingResult,
    ProviderDocument,
    ProviderHealth,
    ProviderHealthStatus,
    ProviderKind,
    ProviderMetadata,
)


class _FakeProvider:
    provider_name = "fake"
    provider_version = "fake-v1"
    provider_kind = ProviderKind.OCR
    configuration: dict[str, Any] = {}

    async def initialize(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            provider_name=self.provider_name,
            provider_version=self.provider_version,
            provider_kind=self.provider_kind,
            supported_file_types=("pdf", "png", "jpeg", "tiff", "bmp", "webp", "heic"),
            supported_document_types=("scanned_pdf", "printed_image", "handwritten_image"),
            configuration=self.configuration,
            startup_timestamp=datetime.now(UTC),
        )

    def health(self) -> ProviderHealth:
        timestamp = datetime.now(UTC)
        return ProviderHealth(
            provider_name=self.provider_name,
            status=ProviderHealthStatus.READY,
            checked_at=timestamp,
            startup_timestamp=timestamp,
            detail="Synthetic provider is ready.",
        )


class FakeOCRProvider(_FakeProvider, BaseOCRProvider):
    provider_name = "qwen3-vl"
    provider_kind = ProviderKind.OCR
    configuration = {
        "model_name": "synthetic-qwen3-vl",
        "model_revision": "synthetic-revision",
        "prompt_version": "synthetic-prompt-v1",
        "confidence_calibration_version": "synthetic-calibration-v1",
    }

    def __init__(self) -> None:
        self.calls = 0

    def process(self, document: ProviderDocument) -> OCRProviderResult:
        del document
        self.calls += 1
        return OCRProviderResult(
            text="BP 120 / 80 haemoglobn 6 . 5 %",
            document_type="scanned_pdf",
            confidence=0.93,
            confidence_method="mean_generated_token_probability",
            processing_time_ms=12.5,
        )


class FakePostProcessor(_FakeProvider, BasePostProcessor):
    provider_name = "symspell"
    provider_kind = ProviderKind.POSTPROCESSOR
    configuration = {"rule_version": "synthetic-rules-v1"}

    def normalize(self, text: str, *, document_type: str) -> PostProcessingResult:
        del text, document_type
        return PostProcessingResult(
            normalized_text="blood pressure (BP) 120 / 80 hemoglobin 6.5%",
            processing_time_ms=1.5,
        )

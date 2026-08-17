"""Build reproducible OCR result records from provider outputs."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from app.ocr.domain.records import OCRResultRecord, OCRReviewState
from app.ocr.providers.contracts import (
    OCRProviderResult,
    PostProcessingResult,
    ProviderMetadata,
)


@dataclass(frozen=True, slots=True)
class OCRResultBuilder:
    pipeline_version: str
    schema_version: str

    def build(
        self,
        *,
        request_id: UUID,
        report_id: UUID,
        ocr_id: UUID,
        ocr_result: OCRProviderResult,
        postprocessed: PostProcessingResult,
        ocr_metadata: ProviderMetadata,
        postprocessor_metadata: ProviderMetadata,
        processing_time_ms: float,
        cache_hit: bool,
        metadata: Mapping[str, Any],
        upload_time_ms: float | None = None,
    ) -> OCRResultRecord:
        warnings = tuple(dict.fromkeys((*ocr_result.warnings, *postprocessed.warnings)))
        review_required = bool(warnings) or ocr_result.document_type == "unknown"
        ocr_configuration = ocr_metadata.configuration
        postprocessor_configuration = postprocessor_metadata.configuration
        return OCRResultRecord(
            request_id=request_id,
            report_id=report_id,
            process_id=ocr_id,
            raw_text=ocr_result.text,
            normalized_text=postprocessed.normalized_text,
            document_type=ocr_result.document_type,
            page_count=int(metadata["page_count"]),
            provider_name=ocr_metadata.provider_name,
            model_name=str(ocr_configuration.get("model_name", ocr_metadata.provider_name)),
            model_version=str(
                ocr_configuration.get("model_revision", ocr_metadata.provider_version)
            ),
            provider_version=ocr_metadata.provider_version,
            pipeline_version=self.pipeline_version,
            rule_version=str(
                postprocessor_configuration.get(
                    "rule_version", postprocessor_metadata.provider_version
                )
            ),
            schema_version=self.schema_version,
            confidence=ocr_result.confidence,
            confidence_method=ocr_result.confidence_method,
            calibration_version=(
                str(ocr_configuration["confidence_calibration_version"])
                if ocr_result.confidence is not None
                else None
            ),
            processing_time_ms=processing_time_ms,
            cache_hit=cache_hit,
            review_state=(
                OCRReviewState.REQUIRED if review_required else OCRReviewState.NOT_REQUIRED
            ),
            warnings=warnings,
            metadata={
                **dict(metadata),
                "ocr": _provider_identity(ocr_metadata),
                "postprocessor": _provider_identity(postprocessor_metadata),
                "document_type_source": (
                    "native_pdf_text_layer"
                    if ocr_result.document_type == "digital_pdf"
                    else "qwen3-vl_prompt"
                ),
                "upload_time_ms": upload_time_ms,
                "decode_time_ms": _decode_time_ms(ocr_result),
                "ocr_processing_time_ms": ocr_result.processing_time_ms,
                "postprocessing_time_ms": postprocessed.processing_time_ms,
            },
            created_at=datetime.now(UTC),
        )


def _decode_time_ms(ocr_result: OCRProviderResult) -> float | None:
    """Return the provider-reported decode/render time when available.

    Not every ``BaseOCRProvider`` implementation measures this separately, so
    it is read defensively rather than added to the provider-neutral contract.
    """

    statistics = getattr(ocr_result, "statistics", None)
    return getattr(statistics, "decode_time_ms", None)


def _provider_identity(metadata: ProviderMetadata) -> dict[str, Any]:
    configuration = metadata.configuration
    return {
        "provider_name": metadata.provider_name,
        "provider_version": metadata.provider_version,
        "model_name": configuration.get("model_name"),
        "model_revision": configuration.get("model_revision"),
    }

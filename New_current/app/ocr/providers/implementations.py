"""Production Qwen3-VL OCR and deterministic medical post-processing providers."""

import asyncio
import gc
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from time import monotonic, perf_counter
from typing import Any

from app.ocr.postprocessing.stages import (
    MedicalAbbreviationStage,
    RegexNormalizationStage,
    StageResult,
    SymSpellStage,
)
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
from app.ocr.providers.documents import (
    DecodedPage,
    DigitalPDFText,
    decode_document_pages,
    extract_digital_pdf_text,
)
from app.ocr.providers.errors import (
    InferenceError,
    ProviderConfigurationError,
    ProviderInitializationError,
    ProviderUnavailableError,
    UnsupportedDocumentError,
)
from app.ocr.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)

_SUPPORTED_FILE_TYPES = (
    "pdf",
    "png",
    "jpeg",
    "jpg",
    "tiff",
    "bmp",
    "webp",
    "heic",
    "heif",
)
_SUPPORTED_DOCUMENT_TYPES = (
    "digital_pdf",
    "scanned_pdf",
    "printed_image",
    "handwritten_image",
    "mixed_document",
    "unknown",
)
_QWEN_OUTPUT_SCHEMA_VERSION = "qwen-ocr-json-v1"
_SENSITIVE_CONFIGURATION_FRAGMENTS = ("credential", "key", "password", "secret", "token")
_QWEN_PROCESSOR_LOAD_LOCK = RLock()


@dataclass(frozen=True, slots=True)
class OCRPageMetadata:
    page_number: int
    width: int
    height: int
    confidence: float | None
    latency_ms: float
    generated_tokens: int


@dataclass(frozen=True, slots=True)
class OCRProcessingStatistics:
    page_count: int
    batch_count: int
    pages_per_second: float
    model_loading_time_ms: float
    gpu_memory_allocated_mb: float | None
    gpu_peak_memory_mb: float | None
    resolved_device: str
    cpu_fallback: bool
    decode_time_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class Qwen3VLOCRResult(OCRProviderResult):
    pages: tuple[OCRPageMetadata, ...] = ()
    statistics: OCRProcessingStatistics | None = None
    provider_metadata: ProviderMetadata | None = None


@dataclass(frozen=True, slots=True)
class PostProcessingStageStatistics:
    stage: str
    latency_ms: float
    corrections: int
    configuration: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class MedicalPostProcessingResult(PostProcessingResult):
    stages: tuple[PostProcessingStageStatistics, ...] = ()
    total_corrections: int = 0


class _ManagedProvider(ABC):
    provider_name: str
    provider_kind: ProviderKind
    supported_file_types = _SUPPORTED_FILE_TYPES
    supported_document_types = _SUPPORTED_DOCUMENT_TYPES

    def __init__(self, configuration: Mapping[str, str]) -> None:
        self._configuration = dict(configuration)
        self._status = ProviderHealthStatus.NEW
        self._startup_timestamp: datetime | None = None
        self._health_detail = "Provider has not been initialized."
        self._lifecycle_lock = RLock()
        self._inference_lock = RLock()
        self._loaded = False
        self._model_loading_time_ms = 0.0
        self._resolved_device = "not_resolved"
        self._cpu_fallback = False
        self._gpu_name: str | None = None
        self._request_count = 0

    @property
    def provider_version(self) -> str:
        return self._required("provider_version")

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    def _initialize_sync(self) -> None:
        with self._lifecycle_lock:
            if self._loaded:
                return
            self._status = ProviderHealthStatus.INITIALIZING
            logger.info(
                "OCR provider initialization started",
                extra=self._log_context("provider_initializing"),
            )
            started = perf_counter()
            try:
                self._validate_configuration()
                self._load_runtime()
            except ProviderConfigurationError:
                self._cleanup_failed_runtime()
                self._status = ProviderHealthStatus.UNAVAILABLE
                self._health_detail = "Provider configuration is invalid."
                logger.exception(
                    "OCR provider configuration rejected",
                    extra=self._log_context("provider_configuration_invalid"),
                )
                raise
            except ProviderInitializationError:
                self._cleanup_failed_runtime()
                self._status = ProviderHealthStatus.UNAVAILABLE
                self._health_detail = "Provider model artifacts are unavailable."
                raise
            except ProviderUnavailableError:
                self._cleanup_failed_runtime()
                self._status = ProviderHealthStatus.UNAVAILABLE
                self._health_detail = "Required provider runtime resources are unavailable."
                raise
            except Exception as exc:
                self._cleanup_failed_runtime()
                self._status = ProviderHealthStatus.UNAVAILABLE
                self._health_detail = "Provider initialization failed."
                logger.exception(
                    "OCR provider initialization failed",
                    extra=self._log_context("provider_initialization_failed"),
                )
                raise ProviderInitializationError(
                    f"{self.provider_name} could not be initialized."
                ) from exc
            self._model_loading_time_ms = round((perf_counter() - started) * 1000, 3)
            self._loaded = True
            self._startup_timestamp = datetime.now(UTC)
            self._status = ProviderHealthStatus.READY
            self._health_detail = "Provider model and configuration are ready."
            logger.info(
                "OCR provider initialized",
                extra={
                    **self._log_context("provider_initialized"),
                    "model_loading_time_ms": self._model_loading_time_ms,
                    "resolved_device": self._resolved_device,
                    "cpu_fallback": self._cpu_fallback,
                    "gpu_name": self._gpu_name,
                },
            )

    async def shutdown(self) -> None:
        await asyncio.to_thread(self._shutdown_sync)

    def _shutdown_sync(self) -> None:
        with self._lifecycle_lock:
            if self._loaded:
                self._release_runtime()
            self._loaded = False
            self._status = ProviderHealthStatus.STOPPED
            self._health_detail = "Provider has been shut down and runtime memory released."
            logger.info(
                "OCR provider shut down",
                extra=self._log_context("provider_stopped"),
            )

    def metadata(self) -> ProviderMetadata:
        runtime_configuration = {
            **self._safe_configuration(),
            "resolved_device": self._resolved_device,
            "cpu_fallback": self._cpu_fallback,
            "gpu_name": self._gpu_name,
            "model_loading_time_ms": self._model_loading_time_ms,
            "warm": self._loaded,
            "request_count": self._request_count,
        }
        return ProviderMetadata(
            provider_name=self.provider_name,
            provider_version=self.provider_version,
            provider_kind=self.provider_kind,
            supported_file_types=self.supported_file_types,
            supported_document_types=self.supported_document_types,
            configuration=runtime_configuration,
            startup_timestamp=self._startup_timestamp,
        )

    def health(self) -> ProviderHealth:
        health = ProviderHealth(
            provider_name=self.provider_name,
            status=self._status,
            checked_at=datetime.now(UTC),
            startup_timestamp=self._startup_timestamp,
            detail=self._health_detail,
        )
        logger.info(
            "OCR provider health checked",
            extra={
                **self._log_context("provider_health_checked"),
                "provider_health": health.status.value,
            },
        )
        return health

    def _validate_configuration(self) -> None:
        for key in self._required_configuration_keys():
            self._required(key)

    def _required_configuration_keys(self) -> tuple[str, ...]:
        return ("provider_version",)

    @abstractmethod
    def _load_runtime(self) -> None: ...

    @abstractmethod
    def _release_runtime(self) -> None: ...

    def _require_ready(self) -> None:
        if not self._loaded or self._status is not ProviderHealthStatus.READY:
            raise ProviderUnavailableError(
                f"{self.provider_name} is not initialized and ready for inference."
            )

    def _cleanup_failed_runtime(self) -> None:
        try:
            self._release_runtime()
        except Exception:
            logger.exception(
                "Partially initialized OCR provider could not release all resources",
                extra=self._log_context("provider_partial_cleanup_failed"),
            )

    def _required(self, key: str) -> str:
        value = self._configuration.get(key, "").strip()
        if not value:
            raise ProviderConfigurationError(
                f"{self.provider_name} requires configuration value {key!r}."
            )
        return value

    def _integer(self, key: str, *, minimum: int, maximum: int | None = None) -> int:
        raw_value = self._required(key)
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ProviderConfigurationError(f"{key} must be an integer.") from exc
        if value < minimum or (maximum is not None and value > maximum):
            upper = f" and at most {maximum}" if maximum is not None else ""
            raise ProviderConfigurationError(f"{key} must be at least {minimum}{upper}.")
        return value

    def _float(self, key: str, *, minimum: float, maximum: float | None = None) -> float:
        raw_value = self._required(key)
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ProviderConfigurationError(f"{key} must be numeric.") from exc
        if value < minimum or (maximum is not None and value > maximum):
            upper = f" and at most {maximum}" if maximum is not None else ""
            raise ProviderConfigurationError(f"{key} must be at least {minimum}{upper}.")
        return value

    def _boolean(self, key: str) -> bool:
        value = self._required(key).casefold()
        if value not in {"true", "false", "1", "0", "yes", "no"}:
            raise ProviderConfigurationError(f"{key} must be a boolean.")
        return value in {"true", "1", "yes"}

    def _safe_configuration(self) -> dict[str, Any]:
        return {
            key: (
                "[REDACTED]"
                if key.casefold() == "prompt"
                or any(
                    fragment in key.casefold() for fragment in _SENSITIVE_CONFIGURATION_FRAGMENTS
                )
                else value
            )
            for key, value in sorted(self._configuration.items())
        }

    def _log_context(self, event: str) -> dict[str, str]:
        return {
            "event": event,
            "provider_kind": self.provider_kind.value,
            "provider_name": self.provider_name,
            "provider_version": self._configuration.get("provider_version", "unconfigured"),
        }


class _TorchModelProvider(_ManagedProvider):
    _torch: Any = None
    _model: Any = None
    _processor: Any = None

    def _resolve_torch_device(self) -> None:
        try:
            import torch
        except (ImportError, ModuleNotFoundError) as exc:
            raise ProviderInitializationError(
                f"{self.provider_name} requires the configured PyTorch dependency."
            ) from exc
        requested_device = self._required("device").casefold()
        allow_cpu_fallback = self._boolean("allow_cpu_fallback")
        if requested_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested_device == "cpu":
            resolved_device = "cpu"
        elif requested_device == "cuda" or requested_device.startswith("cuda:"):
            if torch.cuda.is_available():
                resolved_device = requested_device
            elif allow_cpu_fallback:
                resolved_device = "cpu"
                self._cpu_fallback = True
                logger.warning(
                    "CUDA unavailable; OCR provider using configured CPU fallback",
                    extra={
                        **self._log_context("provider_cpu_fallback"),
                        "requested_device": requested_device,
                        "resolved_device": resolved_device,
                    },
                )
            else:
                raise ProviderUnavailableError(
                    f"CUDA was requested for {self.provider_name} but is unavailable."
                )
        else:
            raise ProviderConfigurationError(
                "device must be 'auto', 'cpu', 'cuda', or a numbered CUDA device."
            )
        self._torch = torch
        self._resolved_device = resolved_device
        self._gpu_name = (
            torch.cuda.get_device_name(torch.device(resolved_device))
            if resolved_device.startswith("cuda")
            else None
        )

    def _release_runtime(self) -> None:
        self._model = None
        self._processor = None
        gc.collect()
        if self._torch is not None and self._resolved_device.startswith("cuda"):
            self._torch.cuda.empty_cache()

    def _move_inputs(self, inputs: Any) -> Any:
        if hasattr(inputs, "to"):
            return inputs.to(self._resolved_device)
        if isinstance(inputs, dict):
            return {
                key: value.to(self._resolved_device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        return inputs

    def _reset_gpu_peak(self) -> None:
        if self._resolved_device.startswith("cuda"):
            self._torch.cuda.reset_peak_memory_stats(self._resolved_device)

    def _gpu_memory(self) -> tuple[float | None, float | None]:
        if not self._resolved_device.startswith("cuda"):
            return None, None
        divisor = 1024 * 1024
        return (
            round(self._torch.cuda.memory_allocated(self._resolved_device) / divisor, 3),
            round(self._torch.cuda.max_memory_allocated(self._resolved_device) / divisor, 3),
        )

    def _raise_inference_error(self, exc: Exception, operation: str) -> None:
        out_of_memory_type = getattr(self._torch, "OutOfMemoryError", MemoryError)
        if (
            isinstance(exc, (MemoryError, out_of_memory_type))
            or "out of memory" in str(exc).casefold()
        ):
            if self._resolved_device.startswith("cuda"):
                self._torch.cuda.empty_cache()
            raise ProviderUnavailableError(
                f"{self.provider_name} exhausted available model memory during {operation}."
            ) from exc
        if isinstance(exc, TimeoutError):
            raise ProviderUnavailableError(
                f"{self.provider_name} exceeded its configured inference timeout."
            ) from exc
        raise InferenceError(f"{self.provider_name} failed during {operation}.") from exc


def _load_qwen_image_only_processor(
    auto_processor: Any,
    processor_type: Any,
    source: str,
    configuration: Mapping[str, Any],
    *,
    torchvision_available: bool,
) -> Any:
    """Load the Qwen processor without its unused video dependency when unavailable."""

    if torchvision_available:
        return auto_processor.from_pretrained(source, **configuration)
    with _QWEN_PROCESSOR_LOAD_LOCK:
        original_attributes = processor_type.attributes
        processor_type.attributes = [
            attribute for attribute in original_attributes if attribute != "video_processor"
        ]
        try:
            return auto_processor.from_pretrained(
                source,
                use_fast=False,
                **configuration,
            )
        finally:
            processor_type.attributes = original_attributes


class Qwen3VLOCRProvider(_TorchModelProvider, BaseOCRProvider):
    provider_name = "qwen3-vl"
    provider_kind = ProviderKind.OCR

    def __init__(self, configuration: Mapping[str, str]) -> None:
        versioned_configuration = {
            **configuration,
            "output_schema_version": _QWEN_OUTPUT_SCHEMA_VERSION,
        }
        super().__init__(versioned_configuration)

    def _required_configuration_keys(self) -> tuple[str, ...]:
        return (
            "provider_version",
            "model_name",
            "model_revision",
            "hf_cache_dir",
            "local_files_only",
            "device",
            "allow_cpu_fallback",
            "dtype",
            "batch_size",
            "max_image_size",
            "max_pages",
            "pdf_render_dpi",
            "timeout_seconds",
            "max_new_tokens",
            "do_sample",
            "temperature",
            "top_p",
            "num_beams",
            "prompt",
            "prompt_version",
            "confidence_threshold",
            "confidence_calibration_version",
            "seed",
        )

    def _validate_configuration(self) -> None:
        super()._validate_configuration()
        self._boolean("local_files_only")
        self._boolean("allow_cpu_fallback")
        self._boolean("do_sample")
        self._integer("batch_size", minimum=1)
        self._integer("max_image_size", minimum=256)
        self._integer("max_pages", minimum=1)
        self._integer("pdf_render_dpi", minimum=72, maximum=600)
        self._float("timeout_seconds", minimum=0.001)
        self._integer("max_new_tokens", minimum=1)
        self._float("temperature", minimum=0.0001)
        self._float("top_p", minimum=0.0001, maximum=1)
        self._integer("num_beams", minimum=1)
        self._integer("seed", minimum=0)
        self._float("confidence_threshold", minimum=0, maximum=1)
        if self._required("dtype").casefold() not in {
            "auto",
            "float32",
            "float16",
            "bfloat16",
        }:
            raise ProviderConfigurationError("dtype must be auto, float32, float16, or bfloat16.")
        if "transcription_only" in self._configuration:
            self._boolean("transcription_only")

    def _load_runtime(self) -> None:
        self._resolve_torch_device()
        try:
            from transformers import (
                AutoProcessor,
                Qwen3VLForConditionalGeneration,
                Qwen3VLProcessor,
            )
            from transformers.utils import is_torchvision_available

            dtype = self._resolve_dtype()
            source, common = self._pretrained_source_and_configuration()
            self._processor = _load_qwen_image_only_processor(
                AutoProcessor,
                Qwen3VLProcessor,
                source,
                common,
                torchvision_available=is_torchvision_available(),
            )
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                source,
                dtype=dtype,
                **common,
            )
            self._model.to(self._resolved_device)
            self._model.eval()
        except ProviderConfigurationError:
            raise
        except Exception as exc:
            logger.exception(
                "Qwen3-VL model loading failed",
                extra=self._log_context("model_loading_failed"),
            )
            raise ProviderInitializationError(
                "The configured Qwen3-VL checkpoint or processor could not be loaded."
            ) from exc

    def _pretrained_source_and_configuration(self) -> tuple[str, dict[str, Any]]:
        revision = self._required("model_revision")
        cache_directory = Path(self._required("hf_cache_dir")).resolve()
        local_files_only = self._boolean("local_files_only")
        if (cache_directory / "config.json").is_file():
            revision_record = (
                cache_directory / ".cache" / "huggingface" / "trees" / f"{revision}.json"
            )
            if not revision_record.is_file():
                raise ProviderConfigurationError(
                    "The local Qwen3-VL checkpoint does not contain provenance for the "
                    "configured immutable revision."
                )
            return str(cache_directory), {"local_files_only": True}
        return self._required("model_name"), {
            "revision": revision,
            "cache_dir": str(cache_directory),
            "local_files_only": local_files_only,
        }

    def _resolve_dtype(self) -> Any:
        configured = self._required("dtype").casefold()
        if self._resolved_device == "cpu":
            if configured in {"auto", "float32"}:
                return self._torch.float32
            if configured == "float16":
                raise ProviderConfigurationError(
                    "float16 is not supported for Qwen3-VL CPU initialization; use "
                    "float32 or bfloat16."
                )
            return self._torch.bfloat16
        if configured == "auto":
            return (
                self._torch.bfloat16
                if self._torch.cuda.is_bf16_supported()
                else self._torch.float16
            )
        return getattr(self._torch, configured)

    def _digital_pdf_result(
        self,
        digital_pdf: DigitalPDFText,
        started: float,
    ) -> Qwen3VLOCRResult:
        elapsed_ms = round((perf_counter() - started) * 1000, 3)
        per_page_latency = round(elapsed_ms / len(digital_pdf.page_texts), 3)
        pages = tuple(
            OCRPageMetadata(
                page_number=index,
                width=width,
                height=height,
                confidence=None,
                latency_ms=per_page_latency,
                generated_tokens=0,
            )
            for index, (width, height) in enumerate(digital_pdf.dimensions, start=1)
        )
        pages_per_second = round(len(digital_pdf.page_texts) / max(elapsed_ms / 1000, 1e-9), 3)
        logger.info(
            "Native digital PDF text extraction completed",
            extra={
                **self._log_context("digital_pdf_extraction_completed"),
                "inference_latency_ms": elapsed_ms,
                "page_count": len(pages),
                "pages_per_second": pages_per_second,
                "resolved_device": "cpu",
            },
        )
        return Qwen3VLOCRResult(
            text=digital_pdf.text,
            document_type="digital_pdf",
            confidence=None,
            confidence_method=None,
            processing_time_ms=elapsed_ms,
            warnings=(),
            pages=pages,
            statistics=OCRProcessingStatistics(
                page_count=len(pages),
                batch_count=0,
                pages_per_second=pages_per_second,
                model_loading_time_ms=self._model_loading_time_ms,
                gpu_memory_allocated_mb=0.0,
                gpu_peak_memory_mb=0.0,
                resolved_device="cpu",
                cpu_fallback=False,
                decode_time_ms=elapsed_ms,
            ),
            provider_metadata=self.metadata(),
        )

    def process(self, document: ProviderDocument) -> Qwen3VLOCRResult:
        return self._process(document, transcription_only=self._transcription_only())

    def process_transcription_only(self, document: ProviderDocument) -> Qwen3VLOCRResult:
        """Run the isolated OCR-output-contract benchmark on the loaded runtime.

        This method is intentionally outside ``BaseOCRProvider`` and is consumed only by
        the benchmark harness. Public OCR routes always call ``process``.
        """

        return self._process(document, transcription_only=True)

    def _process(
        self,
        document: ProviderDocument,
        *,
        transcription_only: bool,
    ) -> Qwen3VLOCRResult:
        self._require_ready()
        self._request_count += 1
        started = perf_counter()
        max_pages = self._integer("max_pages", minimum=1)
        digital_pdf = extract_digital_pdf_text(document, max_pages=max_pages)
        if digital_pdf is not None:
            return self._digital_pdf_result(digital_pdf, started)
        decode_started = perf_counter()
        pages = decode_document_pages(
            document,
            pdf_render_dpi=self._integer("pdf_render_dpi", minimum=72, maximum=600),
            max_image_size=self._integer("max_image_size", minimum=256),
            max_pages=max_pages,
        )
        decode_time_ms = round((perf_counter() - decode_started) * 1000, 3)
        timeout_seconds = self._float("timeout_seconds", minimum=0.001)
        deadline = monotonic() + timeout_seconds
        batch_size = self._integer("batch_size", minimum=1)
        self._reset_gpu_peak()
        page_texts: list[str] = []
        page_document_types: list[str] = []
        page_metadata: list[OCRPageMetadata] = []
        batch_count = 0
        try:
            with self._inference_lock, self._torch.inference_mode():
                for offset in range(0, len(pages), batch_size):
                    _check_deadline(deadline)
                    batch = pages[offset : offset + batch_size]
                    batch_count += 1
                    (
                        texts,
                        document_types,
                        confidences,
                        token_counts,
                        batch_latency_ms,
                    ) = self._process_batch(batch, transcription_only=transcription_only)
                    per_page_latency = round(batch_latency_ms / len(batch), 3)
                    for page, text, document_type, confidence, token_count in zip(
                        batch,
                        texts,
                        document_types,
                        confidences,
                        token_counts,
                        strict=True,
                    ):
                        if not text.strip():
                            raise InferenceError(
                                f"Qwen3-VL returned no OCR text for page {page.page_number}."
                            )
                        page_texts.append(text.strip())
                        page_document_types.append(document_type)
                        page_metadata.append(
                            OCRPageMetadata(
                                page_number=page.page_number,
                                width=page.image.width,
                                height=page.image.height,
                                confidence=confidence,
                                latency_ms=per_page_latency,
                                generated_tokens=token_count,
                            )
                        )
                    _check_deadline(deadline)
        except (UnsupportedDocumentError, ProviderUnavailableError, InferenceError):
            raise
        except Exception as exc:
            self._raise_inference_error(exc, "OCR inference")
        elapsed_ms = round((perf_counter() - started) * 1000, 3)
        confidence = (
            sum(value for value in (page.confidence for page in page_metadata) if value is not None)
            / sum(page.confidence is not None for page in page_metadata)
            if any(page.confidence is not None for page in page_metadata)
            else None
        )
        confidence_method = "mean_generated_token_probability" if confidence is not None else None
        warnings = []
        threshold = self._float("confidence_threshold", minimum=0, maximum=1)
        if confidence is None:
            warnings.append("The model did not expose token probabilities for confidence scoring.")
        elif confidence < threshold:
            warnings.append("OCR confidence is below the configured manual-review threshold.")
        allocated_mb, peak_mb = self._gpu_memory()
        pages_per_second = round(len(pages) / max(elapsed_ms / 1000, 1e-9), 3)
        logger.info(
            "Qwen3-VL OCR completed",
            extra={
                **self._log_context("ocr_completed"),
                "inference_latency_ms": elapsed_ms,
                "page_count": len(pages),
                "pages_per_second": pages_per_second,
                "confidence": confidence,
                "confidence_method": confidence_method,
                "resolved_device": self._resolved_device,
                "gpu_memory_allocated_mb": allocated_mb,
                "gpu_peak_memory_mb": peak_mb,
                "cpu_fallback": self._cpu_fallback,
            },
        )
        return Qwen3VLOCRResult(
            text="\n\n".join(page_texts),
            document_type=_aggregate_document_type(page_document_types),
            confidence=confidence,
            confidence_method=confidence_method,
            processing_time_ms=elapsed_ms,
            warnings=tuple(warnings),
            pages=tuple(page_metadata),
            statistics=OCRProcessingStatistics(
                page_count=len(pages),
                batch_count=batch_count,
                pages_per_second=pages_per_second,
                model_loading_time_ms=self._model_loading_time_ms,
                gpu_memory_allocated_mb=allocated_mb,
                gpu_peak_memory_mb=peak_mb,
                resolved_device=self._resolved_device,
                cpu_fallback=self._cpu_fallback,
                decode_time_ms=decode_time_ms,
            ),
            provider_metadata=self.metadata(),
        )

    def _process_batch(
        self,
        pages: tuple[DecodedPage, ...],
        *,
        transcription_only: bool,
    ) -> tuple[list[str], list[str], list[float | None], list[int], float]:
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": page.image},
                        {"type": "text", "text": self._generation_prompt(transcription_only)},
                    ],
                }
            ]
            for page in pages
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = self._move_inputs(inputs)
        input_ids = inputs["input_ids"]
        input_length = input_ids.shape[1]
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self._integer("max_new_tokens", minimum=1),
            "do_sample": self._boolean("do_sample"),
            "num_beams": self._integer("num_beams", minimum=1),
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if generate_kwargs["do_sample"]:
            generate_kwargs["temperature"] = self._float("temperature", minimum=0.0001)
            generate_kwargs["top_p"] = self._float("top_p", minimum=0.0001, maximum=1)
        batch_started = perf_counter()
        seed = self._integer("seed", minimum=0)
        self._torch.manual_seed(seed)
        if self._resolved_device.startswith("cuda"):
            self._torch.cuda.manual_seed_all(seed)
        output = self._model.generate(**inputs, **generate_kwargs)
        batch_latency_ms = round((perf_counter() - batch_started) * 1000, 3)
        generated = output.sequences[:, input_length:]
        decoded = [
            value.strip()
            for value in self._processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        ]
        if transcription_only:
            texts = decoded
            document_types = ["unknown"] * len(decoded)
        else:
            parsed = [_parse_qwen_response(value) for value in decoded]
            texts = [item["text"] for item in parsed]
            document_types = [item["document_type"] for item in parsed]
        confidences, token_counts = self._generated_confidences(generated, output.scores)
        return texts, document_types, confidences, token_counts, batch_latency_ms

    def _generation_prompt(self, transcription_only: bool) -> str:
        if transcription_only:
            return (
                "Transcribe all visible text exactly in reading order. "
                "Return only the transcription."
            )
        return self._structured_prompt()

    def _transcription_only(self) -> bool:
        """Return the explicit, benchmark-only output-contract selection."""

        return "transcription_only" in self._configuration and self._boolean("transcription_only")

    def _structured_prompt(self) -> str:
        allowed = ", ".join(_SUPPORTED_DOCUMENT_TYPES)
        return (
            f"{self._required('prompt')}\n"
            "Return only one JSON object with exactly two string fields: "
            '"document_type" and "text". '
            f'"document_type" must be one of: {allowed}. '
            'Place the complete verbatim transcription in "text". Do not use markdown fences.'
        )

    def _generated_confidences(
        self,
        generated_tokens: Any,
        scores: tuple[Any, ...] | list[Any] | None,
    ) -> tuple[list[float | None], list[int]]:
        batch_size = generated_tokens.shape[0]
        if not scores:
            return [None] * batch_size, [0] * batch_size
        pad_token_id = getattr(self._processor.tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self._processor.tokenizer, "eos_token_id", None)
        confidences: list[float | None] = []
        token_counts: list[int] = []
        for batch_index in range(batch_size):
            probabilities = []
            for step, score in enumerate(scores):
                if step >= generated_tokens.shape[1]:
                    break
                token_id = int(generated_tokens[batch_index, step].item())
                if token_id == pad_token_id:
                    break
                probability = self._torch.softmax(score[batch_index], dim=-1)[token_id]
                probabilities.append(float(probability.detach().cpu().item()))
                if token_id == eos_token_id:
                    break
            token_counts.append(len(probabilities))
            confidences.append(sum(probabilities) / len(probabilities) if probabilities else None)
        return confidences, token_counts


class SymSpellPostProcessor(_ManagedProvider, BasePostProcessor):
    provider_name = "symspell"
    provider_kind = ProviderKind.POSTPROCESSOR

    def __init__(self, configuration: Mapping[str, str]) -> None:
        super().__init__(configuration)
        self._regex_stage: RegexNormalizationStage | None = None
        self._abbreviation_stage: MedicalAbbreviationStage | None = None
        self._symspell_stage: SymSpellStage | None = None
        self._resolved_device = "cpu"

    def _required_configuration_keys(self) -> tuple[str, ...]:
        return (
            "provider_version",
            "dictionary_path",
            "dictionary_version",
            "abbreviation_dictionary_path",
            "abbreviation_dictionary_version",
            "protected_terms_path",
            "rule_version",
            "max_edit_distance",
            "prefix_length",
            "min_word_length",
            "min_frequency",
        )

    def _validate_configuration(self) -> None:
        super()._validate_configuration()
        max_edit_distance = self._integer("max_edit_distance", minimum=1, maximum=3)
        prefix_length = self._integer("prefix_length", minimum=2)
        if prefix_length <= max_edit_distance:
            raise ProviderConfigurationError("prefix_length must exceed max_edit_distance.")
        self._integer("min_word_length", minimum=2)
        self._integer("min_frequency", minimum=1)

    def _load_runtime(self) -> None:
        try:
            self._regex_stage = RegexNormalizationStage(rule_version=self._required("rule_version"))
            self._abbreviation_stage = MedicalAbbreviationStage(
                dictionary_path=self._required("abbreviation_dictionary_path"),
                dictionary_version=self._required("abbreviation_dictionary_version"),
            )
            self._symspell_stage = SymSpellStage(
                dictionary_path=self._required("dictionary_path"),
                dictionary_version=self._required("dictionary_version"),
                max_edit_distance=self._integer("max_edit_distance", minimum=1, maximum=3),
                prefix_length=self._integer("prefix_length", minimum=2),
                min_word_length=self._integer("min_word_length", minimum=2),
                min_frequency=self._integer("min_frequency", minimum=1),
                protected_terms=self._abbreviation_stage.protected_terms,
                protected_terms_path=self._required("protected_terms_path"),
            )
        except ProviderConfigurationError:
            raise
        except Exception as exc:
            raise ProviderInitializationError(
                "The configured medical post-processing resources could not be loaded."
            ) from exc

    def _release_runtime(self) -> None:
        self._regex_stage = None
        self._abbreviation_stage = None
        self._symspell_stage = None

    def normalize(self, text: str, *, document_type: str) -> MedicalPostProcessingResult:
        self._require_ready()
        self._request_count += 1
        if not text.strip():
            raise InferenceError("OCR post-processing requires non-empty extracted text.")
        if not document_type.strip():
            raise UnsupportedDocumentError("document_type is required for post-processing.")
        assert self._regex_stage is not None
        assert self._abbreviation_stage is not None
        assert self._symspell_stage is not None
        started = perf_counter()
        with self._inference_lock:
            stage_results = []
            current_text = text
            for stage_name, stage in (
                ("regex", self._regex_stage),
                ("medical_abbreviation_dictionary", self._abbreviation_stage),
                ("symspell", self._symspell_stage),
            ):
                result = stage.process(current_text)
                current_text = result.text
                stage_results.append(_stage_statistics(stage_name, result))
        elapsed_ms = round((perf_counter() - started) * 1000, 3)
        total_corrections = sum(stage.corrections for stage in stage_results)
        logger.info(
            "Medical OCR post-processing completed",
            extra={
                **self._log_context("postprocessing_completed"),
                "processing_time_ms": elapsed_ms,
                "corrections": total_corrections,
                "document_type": document_type,
                "rule_version": self._required("rule_version"),
            },
        )
        return MedicalPostProcessingResult(
            normalized_text=current_text,
            processing_time_ms=elapsed_ms,
            warnings=(),
            stages=tuple(stage_results),
            total_corrections=total_corrections,
        )


def _stage_statistics(name: str, result: StageResult) -> PostProcessingStageStatistics:
    return PostProcessingStageStatistics(
        stage=name,
        latency_ms=result.latency_ms,
        corrections=result.corrections,
        configuration=result.configuration,
    )


def _parse_qwen_response(value: str) -> dict[str, str]:
    candidate = value.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            candidate = "\n".join(lines[1:-1]).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end <= start:
        raise InferenceError("Qwen3-VL returned an invalid OCR response envelope.")
    try:
        payload = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        raise InferenceError("Qwen3-VL returned invalid OCR response JSON.") from exc
    if not isinstance(payload, dict) or set(payload) != {"document_type", "text"}:
        raise InferenceError("Qwen3-VL OCR response does not match the required schema.")
    document_type = payload.get("document_type")
    text = payload.get("text")
    if not isinstance(document_type, str) or document_type not in _SUPPORTED_DOCUMENT_TYPES:
        raise InferenceError("Qwen3-VL returned an unsupported document type.")
    if not isinstance(text, str) or not text.strip():
        raise InferenceError("Qwen3-VL returned empty OCR text.")
    return {"document_type": document_type, "text": text.strip()}


def _aggregate_document_type(document_types: list[str]) -> str:
    known = {value for value in document_types if value != "unknown"}
    if not known:
        return "unknown"
    if len(known) == 1:
        return known.pop()
    return "mixed_document"


def _check_deadline(deadline: float) -> None:
    if monotonic() > deadline:
        raise TimeoutError("Configured provider timeout exceeded.")


def register_builtin_providers(registry: ProviderRegistry) -> None:
    registry.register(ProviderKind.OCR, "qwen3-vl", Qwen3VLOCRProvider)
    registry.register(ProviderKind.POSTPROCESSOR, "symspell", SymSpellPostProcessor)

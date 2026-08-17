"""Pinned, local-only IndicTrans2 English-to-Indic provider."""

import asyncio
import hashlib
import logging
import os
import re
from collections.abc import Callable
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any

from app.translation.contracts import BaseTranslationProvider, TranslationProviderMetadata
from app.translation.errors import TranslationPreservationError, TranslationUnavailableError
from app.translation.manifest import (
    PENDING_APPROVAL,
    load_translation_model_manifest,
)

logger = logging.getLogger(__name__)
SUPPORTED_LANGUAGES = {
    "hin_Deva": "Hindi",
    "ben_Beng": "Bengali",
    "guj_Gujr": "Gujarati",
    "kan_Knda": "Kannada",
    "mal_Mlym": "Malayalam",
    "mar_Deva": "Marathi",
    "ory_Orya": "Odia",
    "pan_Guru": "Punjabi",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
}
_PROTECTED = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b|"
    r"\b\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?\s*(?:%|mg/dL|g/dL|mmol/L|"
    r"mEq/L|U/L|IU/L|mmHg|bpm|mg|mcg|mL|mm|cm)?\b",
    re.IGNORECASE,
)
_PLACEHOLDER_PREFIX = "912347"
_PLACEHOLDER_SUFFIX = "568"
_PRIMARY_WEIGHT_FILENAME = "model.safetensors"


class IndicTrans2Provider(BaseTranslationProvider):
    """Load exactly one approved checkpoint and keep it resident for batched inference."""

    def __init__(self, *, manifest_path: Path | None = None) -> None:
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._lock = RLock()
        self._manifest = load_translation_model_manifest(manifest_path)
        self._device = os.getenv("TRANSLATION_CONFIG__DEVICE", "auto")
        self._max_new_tokens = int(os.getenv("TRANSLATION_CONFIG__MAX_NEW_TOKENS", "256"))
        self._model_name = os.getenv(
            "TRANSLATION_CONFIG__MODEL_ID", self._manifest.repository_id
        )
        self._model_revision = os.getenv(
            "TRANSLATION_CONFIG__MODEL_REVISION", self._manifest.pinned_revision
        )
        raw_path = os.getenv("TRANSLATION_CONFIG__MODEL_PATH", self._manifest.local_cache_path)
        if raw_path == PENDING_APPROVAL:
            self._path = None
        else:
            candidate = Path(raw_path).expanduser()
            self._path = (
                candidate
                if candidate.is_absolute()
                else Path(__file__).resolve().parents[3] / candidate
            )
        module_cache = os.getenv("TRANSLATION_CONFIG__MODULE_CACHE_PATH", "").strip()
        self._module_cache_path = (
            Path(module_cache).expanduser()
            if module_cache
            else Path(__file__).resolve().parents[2] / ".model-cache" / "transformers-modules"
        )
        self._model_loading_time_ms: float | None = None
        self._startup_timestamp: datetime | None = None
        self._request_count = 0
        self._detail = self._manifest.provisioning_message

    async def initialize(self, *, strict: bool = True) -> None:
        try:
            self._validate_configuration()
            await asyncio.to_thread(self._load)
            self._detail = "Approved local IndicTrans2 translation provider is ready."
            logger.info(
                "IndicTrans2 translation provider initialized",
                extra={
                    "event": "translation_provider_initialized",
                    "pipeline_stage": "translation",
                    "model_name": self._model_name,
                    "model_revision": self._model_revision,
                    "device": self._device,
                    "model_loading_time_ms": self._model_loading_time_ms,
                },
            )
        except (ImportError, OSError, RuntimeError, TranslationUnavailableError, ValueError) as exc:
            self._detail = str(exc) or "Approved local IndicTrans2 runtime could not initialize."
            logger.warning(
                "IndicTrans2 translation provider is unavailable",
                extra={
                    "event": "translation_provider_unavailable",
                    "pipeline_stage": "translation",
                    "model_name": self._model_name,
                    "model_revision": self._model_revision,
                    "error_type": type(exc).__name__,
                },
            )
            if strict:
                raise TranslationUnavailableError(self._detail) from exc

    def _validate_configuration(self) -> None:
        if not self._manifest.approved:
            raise TranslationUnavailableError(self._manifest.provisioning_message)
        if self._model_name != self._manifest.repository_id:
            raise TranslationUnavailableError(
                "Translation model ID does not match MODEL_MANIFEST.md."
            )
        if self._model_revision != self._manifest.pinned_revision:
            raise TranslationUnavailableError(
                "Translation revision does not match MODEL_MANIFEST.md."
            )
        if self._path is None or not self._path.is_dir():
            raise TranslationUnavailableError(
                "Approved local IndicTrans2 checkpoint directory is unavailable."
            )
        manifest_path = Path(self._manifest.local_cache_path)
        expected_path = (
            manifest_path
            if manifest_path.is_absolute()
            else Path(__file__).resolve().parents[3] / manifest_path
        )
        if self._path.resolve() != expected_path.resolve():
            raise TranslationUnavailableError(
                "TRANSLATION_CONFIG__MODEL_PATH must be the approved local checkpoint directory."
            )
        if not 1 <= self._max_new_tokens <= 256:
            raise TranslationUnavailableError(
                "TRANSLATION_CONFIG__MAX_NEW_TOKENS must be between 1 and 256."
            )
        if self._manifest.expected_sha256 != PENDING_APPROVAL:
            weight_path = self._path / _PRIMARY_WEIGHT_FILENAME
            if not weight_path.is_file():
                raise TranslationUnavailableError(
                    "Approved IndicTrans2 model.safetensors artifact is unavailable."
                )
            observed = self._file_sha256(weight_path)
            if observed != self._manifest.expected_sha256:
                raise TranslationUnavailableError(
                    "IndicTrans2 model.safetensors checksum does not match manifest."
                )

    def _load(self) -> None:
        self._module_cache_path.mkdir(parents=True, exist_ok=True)
        os.environ["HF_MODULES_CACHE"] = str(self._module_cache_path.resolve())
        import torch
        import transformers.dynamic_module_utils as dynamic_module_utils
        from IndicTransToolkit.processor import IndicProcessor
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        dynamic_module_utils.HF_MODULES_CACHE = str(self._module_cache_path.resolve())

        device = self._resolve_device(torch, self._device)
        dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ) if device.startswith("cuda") else torch.float32
        started = perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self._path), trust_remote_code=True, local_files_only=True
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            str(self._path),
            trust_remote_code=True,
            local_files_only=True,
            use_safetensors=True,
            dtype=dtype,
        ).to(device)
        self._model.eval()
        self._processor = IndicProcessor(inference=True)
        self._torch = torch
        self._device = device
        self._model_loading_time_ms = round((perf_counter() - started) * 1000, 3)
        self._startup_timestamp = datetime.now(UTC)

    @staticmethod
    def _resolve_device(torch_module: Any, requested: str) -> str:
        normalized = requested.strip().casefold()
        if normalized == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        if normalized.startswith("cuda") and not torch_module.cuda.is_available():
            raise TranslationUnavailableError("CUDA was requested but is unavailable.")
        return str(torch_module.device(requested))

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        return self.translate_batch((text,), source_language, target_language)[0]

    def translate_batch(
        self, texts: tuple[str, ...], source_language: str, target_language: str
    ) -> tuple[str, ...]:
        if self._model is None or self._tokenizer is None or self._processor is None:
            raise TranslationUnavailableError(self._detail)
        if source_language != "eng_Latn" or target_language not in SUPPORTED_LANGUAGES:
            raise TranslationUnavailableError("The requested language pair is unsupported.")
        if not texts:
            return ()
        self._request_count += 1
        protected: list[list[str]] = [[] for _ in texts]

        def _guard(index: int) -> Callable[[re.Match[str]], str]:
            def replace(match: re.Match[str]) -> str:
                protected[index].append(match.group(0))
                return self._placeholder_marker(len(protected[index]) - 1)

            return replace

        guarded = [_PROTECTED.sub(_guard(index), text) for index, text in enumerate(texts)]
        with self._lock:
            batch = self._processor.preprocess_batch(
                guarded, src_lang=source_language, tgt_lang=target_language
            )
            encoded = self._tokenizer(
                batch, truncation=True, padding="longest", return_tensors="pt"
            ).to(self._device)
            autocast = (
                self._torch.autocast(
                    device_type="cuda", dtype=next(self._model.parameters()).dtype
                )
                if self._device.startswith("cuda")
                else nullcontext()
            )
            with self._torch.inference_mode(), autocast:
                generated = self._model.generate(
                    **encoded,
                    num_beams=5,
                    max_new_tokens=self._max_new_tokens,
                    use_cache=False,
                )
            decoded = self._tokenizer.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            outputs = self._processor.postprocess_batch(decoded, lang=target_language)
        results: list[str] = []
        for output, values in zip(outputs, protected, strict=True):
            for index, value in enumerate(values):
                key = f"{_PLACEHOLDER_PREFIX}{index}{_PLACEHOLDER_SUFFIX}"
                candidates = list(
                    re.finditer(rf"\[[^\]\r\n]*{re.escape(key)}[^\]\r\n]*\]", output)
                )
                if len(candidates) != 1:
                    raise TranslationPreservationError()
                output = output[: candidates[0].start()] + value + output[candidates[0].end() :]
            results.append(output)
        return tuple(results)

    @staticmethod
    def _placeholder_marker(index: int) -> str:
        return f"[IBH{_PLACEHOLDER_PREFIX}{index}{_PLACEHOLDER_SUFFIX}]"

    def metadata(self) -> TranslationProviderMetadata:
        return TranslationProviderMetadata(
            provider_name=self._manifest.provider,
            model_name=self._model_name,
            model_revision=self._model_revision,
            device=self._device,
            ready=self._model is not None,
            detail=self._detail,
            configuration={
                "local_files_only": True,
                "supported_languages": SUPPORTED_LANGUAGES,
                "preservation_policy": "numeric-unit-bracketed-placeholders-v2",
                "model_loading_time_ms": self._model_loading_time_ms,
                "manifest_approved": self._manifest.approved,
                "checksum_verified": self._manifest.expected_sha256 != PENDING_APPROVAL,
                "primary_weight_artifact": _PRIMARY_WEIGHT_FILENAME,
                "dynamic_module_cache": "repository_local",
                "generation_use_cache": False,
                "max_new_tokens": self._max_new_tokens,
                "device": self._device,
                "warm": self._model is not None,
                "request_count": self._request_count,
                "startup_timestamp": (
                    self._startup_timestamp.isoformat() if self._startup_timestamp else None
                ),
            },
        )

    async def shutdown(self) -> None:
        self._model = None
        self._tokenizer = None
        self._processor = None
        logger.info(
            "IndicTrans2 translation provider shutdown",
            extra={
                "event": "translation_provider_shutdown",
                "pipeline_stage": "translation",
                "model_name": self._model_name,
                "model_revision": self._model_revision,
            },
        )

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

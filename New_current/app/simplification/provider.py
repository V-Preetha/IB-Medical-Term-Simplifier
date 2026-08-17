"""Pinned, local-only Qwen3 medical simplification provider."""

import asyncio
import json
import logging
import os
import re
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any, ClassVar

from app.simplification.contracts import (
    BaseSimplificationProvider,
    MedicalTermExplanation,
    ProviderSimplificationResult,
    SimplificationLevelResult,
    SimplificationProviderMetadata,
)
from app.simplification.errors import SimplificationOutputError, SimplificationUnavailableError

logger = logging.getLogger(__name__)
_PROMPT_PATH = Path(__file__).parent / "prompts" / "medical_report_v2.json"
_LEVELS = ("clinical", "general_public", "child_friendly")
_NUMERIC_STRING = re.compile(r"\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?")


class Qwen3SimplificationProvider(BaseSimplificationProvider):
    """Generate all readability levels with one process-wide local Qwen model."""

    _loaded: ClassVar[dict[tuple[str, str], tuple[Any, Any, Any, float]]] = {}
    _load_lock: ClassVar[RLock] = RLock()

    def __init__(self) -> None:
        manifest = self._manifest_inventory()
        self._model_name = os.getenv(
            "SIMPLIFICATION_CONFIG__MODEL_ID", str(manifest["repository_id"])
        )
        self._revision = os.getenv(
            "SIMPLIFICATION_CONFIG__MODEL_REVISION", str(manifest["pinned_revision"])
        )
        if self._model_name != manifest["repository_id"] or self._revision != manifest[
            "pinned_revision"
        ]:
            raise SimplificationUnavailableError(
                "Simplification model identity does not match MODEL_MANIFEST.md."
            )
        self._prompt = self._load_prompt()
        if self._prompt["prompt_version"] != manifest["prompt_version"]:
            raise SimplificationUnavailableError(
                "Simplification prompt identity does not match MODEL_MANIFEST.md."
            )
        self._model_path = os.getenv("SIMPLIFICATION_CONFIG__MODEL_PATH", "").strip()
        self._requested_device = os.getenv("SIMPLIFICATION_CONFIG__DEVICE", "auto")
        self._max_input_characters = int(
            os.getenv("SIMPLIFICATION_CONFIG__MAX_INPUT_CHARACTERS", "48000")
        )
        self._max_new_tokens = int(
            os.getenv("SIMPLIFICATION_CONFIG__MAX_NEW_TOKENS", "1800")
        )
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._torch: Any | None = None
        self._device = self._requested_device
        self._loading_time_ms: float | None = None
        self._startup_timestamp: datetime | None = None
        self._request_count = 0
        self._detail = "Provider has not been initialized."
        self._inference_lock = RLock()

    async def initialize(self, *, strict: bool = True) -> None:
        if os.getenv("REPORT_QWEN_SIMPLIFIER_ENABLED", "true").casefold() == "false":
            self._detail = "Qwen3 simplification is disabled by deployment configuration."
            if strict:
                raise SimplificationUnavailableError(self._detail)
            return
        try:
            await asyncio.to_thread(self._load_model)
            self._detail = "Approved local Qwen3 simplification provider is ready."
            logger.info(
                "Qwen3 simplification provider initialized",
                extra={
                    "event": "simplification_provider_initialized",
                    "pipeline_stage": "simplification",
                    "model_name": self._model_name,
                    "model_revision": self._revision,
                    "device": self._device,
                    "model_loading_time_ms": self._loading_time_ms,
                    "prompt_version": self._prompt["prompt_version"],
                },
            )
        except (ImportError, RuntimeError, ValueError, OSError) as exc:
            self._detail = f"Approved local Qwen3 artifact is unavailable: {exc}"
            logger.error(
                "Qwen3 simplification provider initialization failed",
                extra={
                    "event": "simplification_provider_initialization_failed",
                    "pipeline_stage": "simplification",
                    "model_name": self._model_name,
                    "model_revision": self._revision,
                    "error_type": type(exc).__name__,
                },
            )
            if strict:
                raise SimplificationUnavailableError(self._detail) from exc

    def _load_model(self) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self._resolve_device(torch, self._requested_device)
        source = self._model_path
        if source:
            snapshot = Path(source).expanduser().resolve()
            if not snapshot.is_dir() or snapshot.name != self._revision:
                raise RuntimeError(
                    "SIMPLIFICATION_CONFIG__MODEL_PATH must be the pinned snapshot directory."
                )
            source = str(snapshot)
        else:
            source = snapshot_download(
                self._model_name,
                revision=self._revision,
                local_files_only=True,
            )
        key = (str(Path(source).resolve()), device)
        with self._load_lock:
            loaded = self._loaded.get(key)
            if loaded is None:
                started = perf_counter()
                tokenizer = AutoTokenizer.from_pretrained(
                    source, revision=self._revision, local_files_only=True
                )
                dtype = torch.float32
                if device.startswith("cuda"):
                    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                model = AutoModelForCausalLM.from_pretrained(
                    source,
                    revision=self._revision,
                    local_files_only=True,
                    dtype=dtype,
                ).to(device)
                model.eval()
                model.generation_config.do_sample = False
                model.generation_config.temperature = None
                model.generation_config.top_p = None
                model.generation_config.top_k = None
                loaded = (model, tokenizer, torch, round((perf_counter() - started) * 1000, 3))
                self._loaded[key] = loaded
            self._model, self._tokenizer, self._torch, self._loading_time_ms = loaded
        self._device = device
        self._startup_timestamp = datetime.now(UTC)

    @staticmethod
    def _resolve_device(torch_module: Any, requested: str) -> str:
        normalized = requested.strip().casefold()
        if normalized == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        if normalized.startswith("cuda") and not torch_module.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return str(torch_module.device(requested))

    def simplify(
        self,
        text: str,
        entities: dict[str, tuple[str, ...]],
        linked_concepts: tuple[dict[str, str], ...] = (),
    ) -> ProviderSimplificationResult:
        if self._model is None or self._tokenizer is None or self._torch is None:
            raise SimplificationUnavailableError(self._detail)
        self._request_count += 1
        facts = {
            label: list(dict.fromkeys(values)) for label, values in entities.items() if values
        }
        user_prompt = self._prompt["user_template"]
        replacements = {
            "{source_report_json}": json.dumps(
                text[: self._max_input_characters], ensure_ascii=False
            ),
            "{entities_json}": json.dumps(facts, ensure_ascii=False, separators=(",", ":")),
            "{linked_concepts_json}": json.dumps(
                linked_concepts, ensure_ascii=False, separators=(",", ":")
            ),
            "{authorized_numeric_json}": json.dumps(
                list(dict.fromkeys(_NUMERIC_STRING.findall(text))),
                separators=(",", ":"),
            ),
        }
        for marker, value in replacements.items():
            user_prompt = user_prompt.replace(marker, value)
        messages = (
            {"role": "system", "content": self._prompt["system"]},
            {"role": "user", "content": user_prompt},
        )
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        with self._inference_lock:
            inputs = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
                self._device
            )
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            started = perf_counter()
            autocast = (
                self._torch.autocast(
                    device_type="cuda", dtype=next(self._model.parameters()).dtype
                )
                if self._device.startswith("cuda")
                else nullcontext()
            )
            with self._torch.inference_mode(), autocast:
                output = self._model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=self._max_new_tokens,
                    use_cache=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            generation_time_ms = round((perf_counter() - started) * 1000, 3)
            generated = output[0][prompt_tokens:]
            output_tokens = int(generated.shape[-1])
            decoded = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        try:
            payload = self._parse_json(decoded)
            levels = tuple(self._parse_level(name, payload.get(name)) for name in _LEVELS)
        except RuntimeError as exc:
            raise SimplificationOutputError(
                "Qwen3 returned an invalid structured simplification."
            ) from exc
        return ProviderSimplificationResult(
            levels=levels,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            generation_time_ms=generation_time_ms,
        )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        start = text.find("{")
        if start < 0:
            raise RuntimeError("Qwen3 did not return a JSON object.")
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError as exc:
            raise RuntimeError("Qwen3 returned malformed structured output.") from exc
        if not isinstance(value, dict):
            raise RuntimeError("Qwen3 output must be a JSON object.")
        return value

    @classmethod
    def _parse_level(cls, name: str, value: Any) -> SimplificationLevelResult:
        if not isinstance(value, dict):
            raise RuntimeError(f"Qwen3 output is missing level '{name}'.")
        report = cls._string(value, "simplified_report")
        terms_value = value.get("medical_terms_explained")
        if not isinstance(terms_value, list):
            raise RuntimeError(f"Qwen3 level '{name}' has invalid term explanations.")
        terms: list[MedicalTermExplanation] = []
        for item in terms_value:
            if not isinstance(item, dict):
                raise RuntimeError(f"Qwen3 level '{name}' has invalid term explanations.")
            terms.append(
                MedicalTermExplanation(cls._string(item, "term"), cls._string(item, "explanation"))
            )
        return SimplificationLevelResult(
            level=name,
            simplified_report=report,
            medical_terms_explained=tuple(terms),
            important_findings=cls._string_list(value, "important_findings"),
            suggested_questions_for_doctor=cls._string_list(
                value, "suggested_questions_for_doctor"
            ),
        )

    @staticmethod
    def _string(value: dict[str, Any], key: str) -> str:
        item = value.get(key)
        if not isinstance(item, str) or not item.strip():
            raise RuntimeError(f"Qwen3 field '{key}' is invalid.")
        return item.strip()

    @classmethod
    def _string_list(cls, value: dict[str, Any], key: str) -> tuple[str, ...]:
        items = value.get(key)
        if not isinstance(items, list):
            raise RuntimeError(f"Qwen3 field '{key}' is invalid.")
        return tuple(dict.fromkeys(cls._nonempty_string(item, key) for item in items))

    @staticmethod
    def _nonempty_string(value: Any, key: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"Qwen3 field '{key}' is invalid.")
        return value.strip()

    def metadata(self) -> SimplificationProviderMetadata:
        return SimplificationProviderMetadata(
            provider_name="qwen3",
            model_name=self._model_name,
            model_revision=self._revision,
            prompt_version=str(self._prompt["prompt_version"]),
            device=self._device,
            ready=self._model is not None,
            detail=self._detail,
            configuration={
                "local_files_only": True,
                "deterministic_generation": True,
                "max_input_characters": self._max_input_characters,
                "max_new_tokens": self._max_new_tokens,
                "warm": self._model is not None,
                "request_count": self._request_count,
                "startup_timestamp": (
                    self._startup_timestamp.isoformat() if self._startup_timestamp else None
                ),
            },
            model_loading_time_ms=self._loading_time_ms,
        )

    async def shutdown(self) -> None:
        logger.info(
            "Qwen3 simplification provider shutdown",
            extra={
                "event": "simplification_provider_shutdown",
                "pipeline_stage": "simplification",
                "model_name": self._model_name,
                "model_revision": self._revision,
            },
        )

    @staticmethod
    def _load_prompt() -> dict[str, str]:
        try:
            payload = json.loads(_PROMPT_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SimplificationUnavailableError(
                "The versioned simplification prompt is unavailable."
            ) from exc
        required = ("prompt_version", "system", "user_template")
        if any(
            not isinstance(payload.get(key), str) or not payload[key].strip()
            for key in required
        ):
            raise SimplificationUnavailableError("The versioned simplification prompt is invalid.")
        return payload

    @staticmethod
    def _manifest_inventory() -> dict[str, Any]:
        manifest_path = Path(__file__).resolve().parents[3] / "MODEL_MANIFEST.md"
        try:
            content = manifest_path.read_text(encoding="utf-8")
            body = content.split("<!-- SIMPLIFICATION_MANIFEST_DATA_START -->", 1)[1].split(
                "<!-- SIMPLIFICATION_MANIFEST_DATA_END -->", 1
            )[0]
            payload = json.loads(body.strip().removeprefix("```json").removesuffix("```").strip())
            return payload["production_model"]
        except (OSError, IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise SimplificationUnavailableError(
                "MODEL_MANIFEST.md has no valid simplification production inventory."
            ) from exc

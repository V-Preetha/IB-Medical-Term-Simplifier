"""Local Qwen report simplification with process-wide model reuse."""

import json
import logging
import os
import re
from collections.abc import Iterator
from contextlib import nullcontext
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, RLock, Thread
from time import perf_counter, process_time
from typing import Any, ClassVar

from app.clinical.models import LabResult, SimplificationSections
from app.performance import PipelineTimings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the clinical explanation stage of IB Health.
Create a patient-friendly explanation of the supplied medical report.

Safety and fidelity rules:
- Use only facts explicitly present in SOURCE EVIDENCE or STRUCTURED FACTS.
- Do not invent diagnoses, values, dates, treatments, prognosis, or follow-up.
- Separate what happened from what a reviewer alleges should have happened.
- Ignore document headings, legal boilerplate, and administrative phrases as clinical facts.
- Explain medical language plainly without changing the medical meaning.
- Preserve every numerical value, unit, date, medication name, and anatomical location.
- Mention the important extracted clinical entities in the simple explanation.
- Recommended follow-up may contain only follow-up explicitly documented in the report.
  If none is documented, use: "No explicit follow-up recommendation was documented."
- Timeline items must contain only explicitly dated or clearly sequenced events.

Return one JSON object and no Markdown, commentary, or analysis. Use exactly these keys:
{
  "executive_summary": "string",
  "key_findings": ["string"],
  "timeline": ["string"],
  "medical_terms_explained": ["Term: plain-language explanation"],
  "simple_explanation": "string",
  "recommended_follow_up": ["string"]
}

Be concise:
- Keep the executive summary to at most two sentences.
- Include only clinically important key findings.
- Do not repeat the same fact in multiple list items.
- Explain only medical terms present in STRUCTURED FACTS.
- Return an empty timeline list when no dated or sequenced event is supplied.
"""
_USER_PROMPT_PREFIX = "STRUCTURED FACTS:\n"
_SOURCE_EVIDENCE_PREFIX = "\n\nSOURCE EVIDENCE:\n"
_PROMPT_SENTINEL = "__IB_HEALTH_DYNAMIC_QWEN_CONTENT__"
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:[ \t]+|\n+)|\n+")
_DATE_OR_SEQUENCE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|"
    r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|"
    r"previously|subsequently|follow[- ]?up|later|then)\b",
    re.IGNORECASE,
)
_FOLLOW_UP = re.compile(
    r"\b(?:recommend(?:ed|ation)?|follow[- ]?up|plan|monitor|repeat|"
    r"continue|return)\b",
    re.IGNORECASE,
)
_NUMBER_WITH_OPTIONAL_UNIT = re.compile(
    r"\b\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?\s*"
    r"(?:%|mg/dL|g/dL|mmol/L|mEq/L|U/L|IU/L|mmHg|bpm|mm|cm|mg|mcg|mL)?\b",
    re.IGNORECASE,
)
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


class _CudaUtilizationSampler:
    """Sample CUDA utilization during generation without a hard NVML dependency."""

    def __init__(self, torch_module: Any, device: str) -> None:
        self._torch = torch_module
        self._device = device
        self._stop = Event()
        self._samples: list[float] = []
        self._thread: Thread | None = None

    def __enter__(self) -> "_CudaUtilizationSampler":
        if self._device.startswith("cuda"):
            self._thread = Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)

    @property
    def average_percent(self) -> float | None:
        if not self._samples:
            return None
        return round(sum(self._samples) / len(self._samples), 1)

    def _sample(self) -> None:
        while not self._stop.wait(0.1):
            try:
                self._samples.append(float(self._torch.cuda.utilization(self._device)))
            except (ModuleNotFoundError, RuntimeError):
                return


class QwenMedicalReportSimplifier:
    """Generate structured explanations with one locally cached Qwen model."""

    _instances: ClassVar[dict[tuple[str, str, int, int, bool], "QwenMedicalReportSimplifier"]] = {}
    _instances_lock: ClassVar[RLock] = RLock()

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        model_id: str,
        device: str,
        max_input_characters: int,
        max_new_tokens: int,
        torch_module: Any | None = None,
        generation_config: Any | None = None,
        attention_backend: str = "eager",
        compile_enabled: bool = False,
        compute_dtype: Any | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self.model_id = model_id
        self.device = device
        self.max_input_characters = max_input_characters
        self.max_new_tokens = max_new_tokens
        self._torch = torch_module
        self._generation_config = generation_config
        self.attention_backend = attention_backend
        self.compile_enabled = compile_enabled
        self.compute_dtype = compute_dtype
        self._prompt_prefix, self._prompt_suffix = self._cache_prompt_template()
        self._inference_lock = RLock()

    @classmethod
    def load_once(
        cls,
        model_id_or_path: str,
        *,
        device: str = "auto",
        max_input_characters: int = 48_000,
        max_new_tokens: int = 900,
        compile_model: bool = True,
    ) -> "QwenMedicalReportSimplifier":
        resolved_device = cls._resolve_device(device)
        key = (
            model_id_or_path,
            resolved_device,
            max_input_characters,
            max_new_tokens,
            compile_model,
        )
        with cls._instances_lock:
            existing = cls._instances.get(key)
            if existing is not None:
                return existing

            started_at = perf_counter()
            try:
                import torch
                from huggingface_hub import snapshot_download
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from transformers.generation.configuration_utils import (
                    CompileConfig,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Qwen simplification requires the project's 'clinical-ner' "
                    "dependency extra (PyTorch and Transformers)."
                ) from exc

            try:
                source = (
                    str(Path(model_id_or_path).resolve())
                    if Path(model_id_or_path).exists()
                    else snapshot_download(model_id_or_path, local_files_only=True)
                )
                tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
                attention_backend = cls._attention_backend(torch, resolved_device)
                compute_dtype = cls._compute_dtype(torch, resolved_device)
                model = AutoModelForCausalLM.from_pretrained(
                    source,
                    local_files_only=True,
                    dtype=compute_dtype,
                    attn_implementation=attention_backend,
                )
                model = model.to(resolved_device)
                model.eval()
                model.config.use_cache = True
                generation_config = deepcopy(model.generation_config)
                generation_config.do_sample = False
                generation_config.temperature = None
                generation_config.top_p = None
                generation_config.top_k = None
                generation_config.num_beams = 1
                generation_config.use_cache = True
                generation_config.max_new_tokens = max_new_tokens
                generation_config.output_attentions = False
                generation_config.output_hidden_states = False
                generation_config.output_scores = False
                generation_config.return_dict_in_generate = False
                cuda_compile = compile_model and resolved_device.startswith("cuda")
                generation_config.cache_implementation = "static" if cuda_compile else "dynamic"
                generation_config.disable_compile = not cuda_compile
                generation_config.compile_config = (
                    CompileConfig(mode="reduce-overhead", fullgraph=False) if cuda_compile else None
                )
                if resolved_device.startswith("cuda"):
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.set_float32_matmul_precision("high")
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    "Qwen model files are unavailable or invalid. "
                    f"Download '{model_id_or_path}' into the Hugging Face cache or "
                    "set REPORT_QWEN_SIMPLIFIER_MODEL to a complete local snapshot."
                ) from exc

            provider = cls(
                model,
                tokenizer,
                model_id=model_id_or_path,
                device=resolved_device,
                max_input_characters=max_input_characters,
                max_new_tokens=max_new_tokens,
                torch_module=torch,
                generation_config=generation_config,
                attention_backend=attention_backend,
                compile_enabled=cuda_compile,
                compute_dtype=compute_dtype,
            )
            cls._instances[key] = provider
            logger.info(
                "Qwen medical report simplifier loaded",
                extra={
                    "qwen_model": model_id_or_path,
                    "qwen_requested_device": device,
                    "qwen_device": resolved_device,
                    "qwen_dtype": str(compute_dtype),
                    "qwen_attention_backend": attention_backend,
                    "qwen_kv_cache": generation_config.cache_implementation,
                    "qwen_torch_compile": cuda_compile,
                    "qwen_cuda_available": torch.cuda.is_available(),
                    "qwen_gpu_name": (
                        torch.cuda.get_device_name(resolved_device)
                        if resolved_device.startswith("cuda")
                        else None
                    ),
                    "qwen_load_time_ms": round(
                        (perf_counter() - started_at) * 1000,
                        3,
                    ),
                },
            )
            return provider

    @staticmethod
    def _resolve_device(requested_device: str) -> str:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for Qwen simplification.") from exc

        requested = requested_device.strip().casefold()
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "Qwen CUDA inference was requested, but this PyTorch runtime "
                "does not have an available CUDA device."
            )
        return str(torch.device(requested_device))

    @staticmethod
    def _compute_dtype(torch_module: Any, device: str) -> Any:
        if not device.startswith("cuda"):
            # BF16 is substantially slower when this CPU only exposes AVX2.
            return torch_module.float32
        if (
            hasattr(torch_module.cuda, "is_bf16_supported")
            and torch_module.cuda.is_bf16_supported()
        ):
            return torch_module.bfloat16
        return torch_module.float16

    @staticmethod
    def _attention_backend(torch_module: Any, device: str) -> str:
        if device.startswith("cuda"):
            try:
                from transformers.utils import is_flash_attn_2_available

                if is_flash_attn_2_available():
                    return "flash_attention_2"
            except ImportError:
                pass
        if hasattr(torch_module.nn.functional, "scaled_dot_product_attention"):
            return "sdpa"
        return "eager"

    def _cache_prompt_template(self) -> tuple[str, str]:
        messages = (
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _PROMPT_SENTINEL},
        )
        rendered = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prefix, sentinel, suffix = rendered.partition(_PROMPT_SENTINEL)
        if not sentinel:
            raise RuntimeError("Qwen chat template did not preserve dynamic content.")
        return prefix, suffix

    def simplify(
        self,
        report_text: str,
        entities: dict[str, tuple[str, ...]],
        lab_results: tuple[LabResult, ...],
        *,
        timings: PipelineTimings | None = None,
    ) -> SimplificationSections:
        started_at = perf_counter()
        prompt_started_at = perf_counter()
        compact_entities = self._deduplicate_entities(entities)
        context = self._build_compact_context(
            report_text,
            compact_entities,
            lab_results,
        )
        facts = {
            "entities": {
                category: list(values) for category, values in compact_entities.items() if values
            },
            "lab_results": [result.to_dict() for result in lab_results],
            "chronological_findings": list(context["chronological_findings"]),
            "important_measurements": list(context["important_measurements"]),
        }
        dynamic_content = (
            _USER_PROMPT_PREFIX
            + json.dumps(facts, ensure_ascii=False, separators=(",", ":"))
            + _SOURCE_EVIDENCE_PREFIX
            + "\n".join(context["supporting_text"])
        )
        prompt = self._prompt_prefix + dynamic_content + self._prompt_suffix
        prompt_construction_ms = (perf_counter() - prompt_started_at) * 1000

        with self._inference_lock:
            tokenizer_started_at = perf_counter()
            model_inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
            tokenizer_ms = (perf_counter() - tokenizer_started_at) * 1000
            prompt_tokens = int(model_inputs["input_ids"].shape[-1])

            transfer_started_at = perf_counter()
            model_inputs = model_inputs.to(self.device)
            if self.device.startswith("cuda"):
                self._torch.cuda.synchronize(self.device)
                self._torch.cuda.reset_peak_memory_stats(self.device)
            transfer_ms = (perf_counter() - transfer_started_at) * 1000

            generation_started_at = perf_counter()
            cpu_started_at = process_time()
            autocast = (
                self._torch.autocast(
                    device_type="cuda",
                    dtype=self.compute_dtype,
                )
                if self.device.startswith("cuda")
                else nullcontext()
            )
            with (
                self._torch.inference_mode(),
                autocast,
                _CudaUtilizationSampler(self._torch, self.device) as gpu_sampler,
            ):
                output_ids = self._model.generate(
                    **model_inputs,
                    generation_config=self._generation_config,
                    use_model_defaults=False,
                )
            if self.device.startswith("cuda"):
                self._torch.cuda.synchronize(self.device)
            generation_ms = (perf_counter() - generation_started_at) * 1000
            cpu_time = process_time() - cpu_started_at
            cpu_utilization = round(
                100 * cpu_time / max(generation_ms / 1000, 1e-9) / max(os.cpu_count() or 1, 1),
                1,
            )
            generated_ids = output_ids[0][model_inputs["input_ids"].shape[-1] :]
            output_tokens = int(generated_ids.shape[-1])

            post_started_at = perf_counter()
            generated_text = self._tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()
            payload = self._parse_json_object(generated_text)
            sections = self._to_sections(payload)
            post_processing_ms = (perf_counter() - post_started_at) * 1000

        gpu_peak_allocated_mb = (
            round(self._torch.cuda.max_memory_allocated(self.device) / 1024**2, 1)
            if self.device.startswith("cuda")
            else 0.0
        )
        gpu_peak_reserved_mb = (
            round(self._torch.cuda.max_memory_reserved(self.device) / 1024**2, 1)
            if self.device.startswith("cuda")
            else 0.0
        )
        total_ms = (perf_counter() - started_at) * 1000
        logger.info(
            "Qwen medical report simplification complete",
            extra={
                "qwen_model": self.model_id,
                "qwen_device": self.device,
                "qwen_dtype": str(self.compute_dtype),
                "qwen_attention_backend": self.attention_backend,
                "qwen_torch_compile": self.compile_enabled,
                "qwen_kv_cache": self._generation_config.cache_implementation,
                "qwen_processing_time_ms": round(total_ms, 3),
                "qwen_prompt_construction_time_ms": round(
                    prompt_construction_ms,
                    3,
                ),
                "qwen_tokenizer_time_ms": round(tokenizer_ms, 3),
                "qwen_device_transfer_time_ms": round(transfer_ms, 3),
                "qwen_generation_time_ms": round(generation_ms, 3),
                "qwen_post_processing_time_ms": round(post_processing_ms, 3),
                "qwen_prompt_tokens": prompt_tokens,
                "qwen_output_tokens": output_tokens,
                "qwen_generation_tokens_per_second": round(
                    output_tokens / max(generation_ms / 1000, 1e-9),
                    3,
                ),
                "qwen_cpu_utilization_percent": cpu_utilization,
                "qwen_gpu_utilization_percent": gpu_sampler.average_percent,
                "qwen_peak_vram_allocated_mb": gpu_peak_allocated_mb,
                "qwen_peak_vram_reserved_mb": gpu_peak_reserved_mb,
                "qwen_source_characters": sum(len(item) for item in context["supporting_text"]),
                "qwen_original_characters": len(report_text),
                "qwen_prompt_reduction_percent": round(
                    100
                    * (
                        1
                        - sum(len(item) for item in context["supporting_text"])
                        / max(1, len(report_text))
                    ),
                    1,
                ),
                "processing_timestamp": datetime.now(UTC).isoformat(),
            },
        )
        if timings is not None:
            timings.record("Qwen Prompt Construction", prompt_construction_ms)
            timings.record("Qwen Tokenization", tokenizer_ms)
            timings.record("Qwen Generation", generation_ms)
            timings.record("Qwen Post-processing", post_processing_ms)
            timings.record("Qwen Simplification", total_ms)
        return sections

    def _build_compact_context(
        self,
        report_text: str,
        entities: dict[str, tuple[str, ...]],
        lab_results: tuple[LabResult, ...],
    ) -> dict[str, tuple[str, ...]]:
        entity_terms = tuple(value.casefold() for values in entities.values() for value in values)
        lab_terms = tuple(result.name.casefold() for result in lab_results)
        evidence: list[str] = []
        chronology: list[str] = []
        measurements: list[str] = [
            f"{result.name}: {result.value} {result.unit}".strip() for result in lab_results
        ]
        measurements.extend(entities.get("measurements", ()))
        measurements.extend(entities.get("vital_signs", ()))

        first_sentences: list[str] = []
        for sentence in self._sentences(report_text):
            if len(first_sentences) < 8:
                first_sentences.append(sentence)
            if not sentence:
                continue
            folded = sentence.casefold()
            is_evidence = (
                any(term in folded for term in entity_terms)
                or any(term in folded for term in lab_terms)
                or bool(_NUMBER_WITH_OPTIONAL_UNIT.search(sentence))
                or bool(_FOLLOW_UP.search(sentence))
            )
            if is_evidence:
                evidence.append(sentence)
            if _DATE_OR_SEQUENCE.search(sentence) and is_evidence:
                chronology.append(sentence)

        unique_evidence = tuple(dict.fromkeys(evidence))
        if not unique_evidence:
            unique_evidence = tuple(first_sentences)
        supporting_text = self._bounded_evidence(unique_evidence)
        supporting_set = set(supporting_text)
        return {
            "supporting_text": supporting_text,
            "chronological_findings": tuple(
                sentence for sentence in dict.fromkeys(chronology) if sentence in supporting_set
            ),
            "important_measurements": tuple(dict.fromkeys(measurements)),
        }

    @staticmethod
    def _sentences(report_text: str) -> Iterator[str]:
        start = 0
        for match in _SENTENCE_BOUNDARY.finditer(report_text):
            sentence = " ".join(report_text[start : match.start()].split())
            if sentence:
                yield sentence
            start = match.end()
        final = " ".join(report_text[start:].split())
        if final:
            yield final

    def _bounded_evidence(self, evidence: tuple[str, ...]) -> tuple[str, ...]:
        limit = min(self.max_input_characters, 12_000)
        output: list[str] = []
        used = 0
        for sentence in evidence:
            required = len(sentence) + (1 if output else 0)
            if output and used + required > limit:
                break
            output.append(sentence[:limit] if not output else sentence)
            used += required
        return tuple(output)

    @staticmethod
    def _deduplicate_entities(
        entities: dict[str, tuple[str, ...]],
    ) -> dict[str, tuple[str, ...]]:
        output: dict[str, tuple[str, ...]] = {}
        for category, values in entities.items():
            ranked = sorted(
                dict.fromkeys(values),
                key=lambda value: (-len(_NON_WORD.sub(" ", value).split()), value.casefold()),
            )
            selected: list[tuple[str, frozenset[str]]] = []
            for value in ranked:
                tokens = frozenset(_NON_WORD.sub(" ", value.casefold()).strip().split())
                if not tokens or any(tokens <= existing for _, existing in selected):
                    continue
                selected.append((value, tokens))
            selected_values = {value for value, _ in selected}
            output[category] = tuple(value for value in values if value in selected_values)
        return output

    @staticmethod
    def _parse_json_object(generated_text: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        start = generated_text.find("{")
        if start < 0:
            raise RuntimeError("Qwen simplification did not return a JSON object.")
        try:
            payload, _ = decoder.raw_decode(generated_text[start:])
        except json.JSONDecodeError as exc:
            raise RuntimeError("Qwen simplification returned malformed structured output.") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Qwen simplification output must be a JSON object.")
        return payload

    @classmethod
    def _to_sections(cls, payload: dict[str, Any]) -> SimplificationSections:
        return SimplificationSections(
            executive_summary=cls._required_string(payload, "executive_summary"),
            important_findings=cls._string_list(payload, "key_findings"),
            timeline=cls._string_list(payload, "timeline"),
            medical_terms=cls._string_list(payload, "medical_terms_explained"),
            simplified_explanation=cls._required_string(
                payload,
                "simple_explanation",
            ),
            recommended_follow_up=cls._string_list(
                payload,
                "recommended_follow_up",
            ),
        )

    @staticmethod
    def _required_string(payload: dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"Qwen simplification field '{key}' is invalid.")
        return value.strip()

    @staticmethod
    def _string_list(payload: dict[str, Any], key: str) -> tuple[str, ...]:
        value = payload.get(key)
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise RuntimeError(f"Qwen simplification field '{key}' is invalid.")
        return tuple(dict.fromkeys(item.strip() for item in value))

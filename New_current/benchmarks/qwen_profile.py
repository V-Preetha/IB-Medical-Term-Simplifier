"""Profile the Qwen-only baseline or optimized generation path."""

import argparse
import json
import os
import sys
from contextlib import nullcontext, suppress
from copy import deepcopy
from pathlib import Path
from threading import Event, Thread
from time import perf_counter, process_time

import fitz
import psutil
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.clinical.qwen_simplifier import (  # noqa: E402
    _SOURCE_EVIDENCE_PREFIX,
    _SYSTEM_PROMPT,
    _USER_PROMPT_PREFIX,
    QwenMedicalReportSimplifier,
)


class ResourceSampler:
    def __init__(self, device: str) -> None:
        self.device = device
        self.peak_ram = 0
        self.cpu_samples: list[float] = []
        self.gpu_samples: list[float] = []
        self._stop = Event()
        self._thread = Thread(target=self._sample, daemon=True)

    def __enter__(self):
        psutil.Process().cpu_percent(interval=None)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()

    def _sample(self) -> None:
        process = psutil.Process()
        while not self._stop.wait(0.1):
            self.peak_ram = max(self.peak_ram, process.memory_info().rss)
            self.cpu_samples.append(process.cpu_percent(interval=None) / psutil.cpu_count())
            if self.device.startswith("cuda"):
                with suppress(ModuleNotFoundError, RuntimeError):
                    self.gpu_samples.append(float(torch.cuda.utilization(self.device)))


def source_text() -> str:
    path = PROJECT_ROOT / "tests" / "OMC Report Sample - Cardio.pdf"
    with fitz.open(path) as document:
        return document[0].get_text().replace("\x02", " ")


def dynamic_content(helper: QwenMedicalReportSimplifier) -> str:
    entities = {
        "diseases": (
            "thoracic aneurysm",
            "abdominal aneurysm",
            "hypertension",
            "hypothyroidism",
            "depression",
        ),
        "symptoms": ("syncope",),
        "anatomy": ("aorta",),
        "procedures": (),
        "measurements": (),
        "laboratory_tests": (),
        "biomarkers": (),
        "medications": (),
        "vital_signs": (),
    }
    context = helper._build_compact_context(source_text(), entities, ())
    facts = {
        "entities": {category: list(values) for category, values in entities.items() if values},
        "lab_results": [],
        "chronological_findings": list(context["chronological_findings"]),
        "important_measurements": list(context["important_measurements"]),
    }
    return (
        _USER_PROMPT_PREFIX
        + json.dumps(facts, ensure_ascii=False, separators=(",", ":"))
        + _SOURCE_EVIDENCE_PREFIX
        + "\n".join(context["supporting_text"])
    )


def profile(mode: str, max_new_tokens: int) -> dict[str, object]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if mode == "optimized":
        helper = QwenMedicalReportSimplifier.load_once(
            "Qwen/Qwen3-0.6B",
            device=device,
            max_new_tokens=max_new_tokens,
        )
        model = helper._model
        tokenizer = helper._tokenizer
        generation_config = deepcopy(helper._generation_config)
        compute_dtype = helper.compute_dtype
        attention = helper.attention_backend
    else:
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        source = snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            source,
            local_files_only=True,
            dtype="auto",
        ).to(device)
        model.eval()
        helper = object.__new__(QwenMedicalReportSimplifier)
        helper.max_input_characters = 48_000
        helper._tokenizer = tokenizer
        helper._prompt_prefix, helper._prompt_suffix = helper._cache_prompt_template()
        generation_config = None
        compute_dtype = model.dtype
        attention = model.config._attn_implementation

    dynamic = dynamic_content(helper)
    prompt_started = perf_counter()
    if mode == "optimized":
        prompt = helper._prompt_prefix + dynamic + helper._prompt_suffix
    else:
        prompt = tokenizer.apply_chat_template(
            (
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": dynamic},
            ),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    prompt_ms = (perf_counter() - prompt_started) * 1000

    tokenizer_started = perf_counter()
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    tokenizer_ms = (perf_counter() - tokenizer_started) * 1000
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    inputs = inputs.to(device)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    cpu_started = process_time()
    generation_started = perf_counter()
    autocast = (
        torch.autocast("cuda", dtype=compute_dtype)
        if device.startswith("cuda") and mode == "optimized"
        else nullcontext()
    )
    with ResourceSampler(device) as resources, torch.inference_mode(), autocast:
        if generation_config is None:
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            output = model.generate(
                **inputs,
                generation_config=generation_config,
                use_model_defaults=False,
            )
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    generation_ms = (perf_counter() - generation_started) * 1000
    cpu_time = process_time() - cpu_started

    output_ids = output[0][prompt_tokens:]
    post_started = perf_counter()
    tokenizer.decode(output_ids, skip_special_tokens=True)
    post_ms = (perf_counter() - post_started) * 1000
    output_tokens = int(output_ids.shape[-1])

    return {
        "mode": mode,
        "device": device,
        "dtype": str(compute_dtype),
        "attention": attention,
        "prompt_construction_ms": round(prompt_ms, 3),
        "tokenizer_ms": round(tokenizer_ms, 3),
        "generation_ms": round(generation_ms, 1),
        "post_processing_ms": round(post_ms, 3),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "generation_tokens_per_second": round(
            output_tokens / (generation_ms / 1000),
            3,
        ),
        "cpu_utilization_percent": round(
            100 * cpu_time / (generation_ms / 1000) / max(os.cpu_count() or 1, 1),
            1,
        ),
        "sampled_cpu_utilization_percent": round(
            sum(resources.cpu_samples) / len(resources.cpu_samples),
            1,
        ),
        "gpu_utilization_percent": (
            round(sum(resources.gpu_samples) / len(resources.gpu_samples), 1)
            if resources.gpu_samples
            else None
        ),
        "peak_ram_mb": round(resources.peak_ram / 1024**2, 1),
        "peak_vram_allocated_mb": (
            round(torch.cuda.max_memory_allocated(device) / 1024**2, 1)
            if device.startswith("cuda")
            else 0.0
        ),
        "peak_vram_reserved_mb": (
            round(torch.cuda.max_memory_reserved(device) / 1024**2, 1)
            if device.startswith("cuda")
            else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("baseline", "optimized"))
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = profile(args.mode, args.max_new_tokens)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

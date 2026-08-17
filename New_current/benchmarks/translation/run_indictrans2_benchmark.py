"""Run offline, reproducible IndicTrans2 translation runtime validation."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import psutil

from app.translation.provider import IndicTrans2Provider


@dataclass(frozen=True, slots=True)
class Measurement:
    language: str
    source: str
    translated: str
    latency_ms: float
    protected_values: tuple[str, ...]
    protected_values_preserved: bool


SAMPLES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "hin_Deva",
        "HbA1c is 8.7%. Metformin 500 mg twice daily.",
        ("8.7%", "500 mg"),
    ),
    (
        "tam_Taml",
        "Blood pressure is 140/90 mmHg. Glucose is 126 mg/dL.",
        ("140/90 mmHg", "126 mg/dL"),
    ),
    (
        "kan_Knda",
        "Stage II breast cancer follow-up is dated 15/10/2024.",
        ("15/10/2024",),
    ),
)


def _measure(
    provider: IndicTrans2Provider,
    language: str,
    text: str,
    protected_values: tuple[str, ...],
) -> Measurement:
    started = perf_counter()
    translated = provider.translate(text, "eng_Latn", language)
    latency_ms = round((perf_counter() - started) * 1000, 3)
    return Measurement(
        language=language,
        source=text,
        translated=translated,
        latency_ms=latency_ms,
        protected_values=protected_values,
        protected_values_preserved=all(value in translated for value in protected_values),
    )


def _markdown(payload: dict[str, object]) -> str:
    rows = payload["measurements"]
    table = "\n".join(
        f"| {row['language']} | {row['latency_ms']} | {row['protected_values_preserved']} |"
        for row in rows
    )
    return f"""# IndicTrans2 Runtime Benchmark

## Runtime

| Field | Value |
| --- | --- |
| Repository | {payload['model']['model_name']} |
| Revision | {payload['model']['model_revision']} |
| Device | {payload['model']['device']} |
| Model loading time (ms) | {payload['model']['configuration']['model_loading_time_ms']} |
| First inference (ms) | {payload['first_inference_ms']} |
| Warm inference (ms) | {payload['warm_inference_ms']} |
| Batch inference (ms) | {payload['batch_inference_ms']} |
| Batch throughput (pages/texts per sec) | {payload['batch_texts_per_second']} |
| Process RSS (MiB) | {payload['process_rss_mib']} |
| Peak GPU allocation (MiB) | {payload['peak_gpu_memory_mib']} |

## Required language checks

| Target | Latency (ms) | Exact protected values preserved |
| --- | ---: | --- |
{table}

The checks above are runtime/model validation with synthetic text. The retained numerical
values prove the provider's fail-closed preservation guard; semantic and clinical
translation quality remain subject to clinical validation.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    provider = IndicTrans2Provider()
    asyncio.run(provider.initialize())
    metadata = provider.metadata()
    torch = provider._torch
    if metadata.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    first = _measure(provider, *SAMPLES[0])
    warm = _measure(provider, *SAMPLES[1])
    language_measurements = (first, warm, _measure(provider, *SAMPLES[2]))
    batch_started = perf_counter()
    batch_result = provider.translate_batch(
        tuple(sample[1] for sample in SAMPLES),
        "eng_Latn",
        "hin_Deva",
    )
    batch_latency_ms = round((perf_counter() - batch_started) * 1000, 3)
    rss_mib = round(psutil.Process().memory_info().rss / (1024 * 1024), 3)
    gpu_mib = (
        round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3)
        if metadata.device.startswith("cuda")
        else None
    )
    payload: dict[str, object] = {
        "benchmark_schema_version": "indictrans2-runtime-benchmark-v1",
        "model": {
            "model_name": metadata.model_name,
            "model_revision": metadata.model_revision,
            "device": metadata.device,
            "configuration": metadata.configuration,
        },
        "first_inference_ms": first.latency_ms,
        "warm_inference_ms": warm.latency_ms,
        "batch_inference_ms": batch_latency_ms,
        "batch_texts_per_second": round(len(batch_result) / (batch_latency_ms / 1000), 3),
        "process_rss_mib": rss_mib,
        "peak_gpu_memory_mib": gpu_mib,
        "measurements": [asdict(row) for row in language_measurements],
        "batch_target_language": "hin_Deva",
        "batch_outputs": list(batch_result),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "indictrans2_benchmark.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "indictrans2_benchmark.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(language_measurements[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in language_measurements)
    (args.output_dir / "indictrans2_benchmark.md").write_text(
        _markdown(payload), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

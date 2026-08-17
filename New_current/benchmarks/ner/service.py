"""Evaluation-only service used by the offline NER benchmark runner."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from threading import Event, Thread
from time import perf_counter
from uuid import UUID, uuid4

import psutil

from app.ner.contracts import BaseNERProvider, NERProviderHealth, NormalizedEntity
from app.ner.providers import NERProviderRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    precision: float | None
    recall: float | None
    f1_score: float | None
    entity_level_accuracy: float | None
    false_positives: int | None
    false_negatives: int | None
    inference_latency_ms: float
    peak_ram_mb: float
    peak_gpu_memory_mb: float | None
    model_loading_time_ms: float | None
    tokens_per_second: float | None


@dataclass(frozen=True, slots=True)
class NERBenchmarkResult:
    request_id: UUID
    model: str
    model_revision: str
    text: str
    entities: tuple[NormalizedEntity, ...]
    metrics: BenchmarkMetrics
    warnings: tuple[str, ...]


class NERBenchmarkService:
    """Select and run candidates only through the offline evaluation registry."""

    def __init__(self, registry: NERProviderRegistry) -> None:
        self._registry = registry
        self._instances: dict[str, BaseNERProvider] = {}
        self._lock = asyncio.Lock()

    async def benchmark(
        self,
        model: str,
        text: str,
        references: tuple[NormalizedEntity, ...] | None = None,
    ) -> NERBenchmarkResult:
        provider = await self._provider(model)
        monitor = _ResourceMonitor()
        monitor.start()
        started = perf_counter()
        try:
            output = await asyncio.to_thread(provider.extract, text)
        finally:
            peak_ram_mb, peak_gpu_mb = monitor.stop()
        latency_ms = round((perf_counter() - started) * 1000, 3)
        precision, recall, f1, accuracy, fp, fn = _score(output.entities, references)
        tokens_per_second = (
            round(output.token_count / (latency_ms / 1000), 3)
            if output.token_count is not None and latency_ms > 0
            else None
        )
        metadata = provider.metadata()
        request_id = uuid4()
        logger.info(
            "NER candidate benchmark completed",
            extra={
                "event": "ner_benchmark_completed",
                "request_id": str(request_id),
                "pipeline_stage": "ner_evaluation",
                "provider_name": metadata.provider_name,
                "model_revision": metadata.model_revision,
                "inference_latency_ms": latency_ms,
                "entity_count": len(output.entities),
                "peak_ram_mb": peak_ram_mb,
                "peak_gpu_memory_mb": peak_gpu_mb,
            },
        )
        return NERBenchmarkResult(
            request_id,
            model,
            metadata.model_revision,
            text,
            output.entities,
            BenchmarkMetrics(
                precision,
                recall,
                f1,
                accuracy,
                fp,
                fn,
                latency_ms,
                peak_ram_mb,
                peak_gpu_mb,
                metadata.loading_time_ms,
                tokens_per_second,
            ),
            output.warnings,
        )

    def models(self) -> tuple[NERProviderHealth, ...]:
        return tuple(self._instance(name).health() for name in self._registry.names())

    async def shutdown(self) -> None:
        for provider in reversed(tuple(self._instances.values())):
            await provider.shutdown()

    async def _provider(self, name: str) -> BaseNERProvider:
        provider = self._instance(name)
        async with self._lock:
            await provider.initialize()
        return provider

    def _instance(self, name: str) -> BaseNERProvider:
        key = name.strip().casefold()
        if key not in self._instances:
            self._instances[key] = self._registry.create(key)
        return self._instances[key]


class _ResourceMonitor:
    def __init__(self) -> None:
        self._stop = Event()
        self._peak = psutil.Process().memory_info().rss
        self._thread = Thread(target=self._sample, daemon=True)
        self._gpu_peak: float | None = None

    def start(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
        self._thread.start()

    def stop(self) -> tuple[float, float | None]:
        self._stop.set()
        self._thread.join(timeout=1)
        self._peak = max(self._peak, psutil.Process().memory_info().rss)
        try:
            import torch

            if torch.cuda.is_available():
                self._gpu_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except ImportError:
            pass
        return round(self._peak / (1024 * 1024), 3), self._gpu_peak

    def _sample(self) -> None:
        process = psutil.Process()
        while not self._stop.wait(0.01):
            self._peak = max(self._peak, process.memory_info().rss)


def _score(
    predicted: tuple[NormalizedEntity, ...],
    references: tuple[NormalizedEntity, ...] | None,
) -> tuple[float | None, float | None, float | None, float | None, int | None, int | None]:
    if references is None:
        return None, None, None, None, None, None
    predicted_keys = {(item.start, item.end, item.label) for item in predicted}
    reference_keys = {(item.start, item.end, item.label) for item in references}
    true_positive = len(predicted_keys & reference_keys)
    false_positive = len(predicted_keys - reference_keys)
    false_negative = len(reference_keys - predicted_keys)
    precision = true_positive / (true_positive + false_positive) if predicted_keys else 0.0
    recall = true_positive / len(reference_keys) if reference_keys else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = true_positive + false_positive + false_negative
    accuracy = true_positive / total if total else 1.0
    return precision, recall, f1, accuracy, false_positive, false_negative

import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional

import psutil
import torch


LOGGER = logging.getLogger(__name__)


class ResourceMonitor:
    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = interval_seconds
        self.process = psutil.Process(os.getpid())
        self.peak_rss_mb = 0.0
        self.peak_cpu_percent = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.process.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return {
            "peak_ram_mb": self.peak_rss_mb,
            "peak_cpu_percent": self.peak_cpu_percent,
        }

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                rss_mb = self.process.memory_info().rss / (1024**2)
                cpu = self.process.cpu_percent(interval=None)
                self.peak_rss_mb = max(self.peak_rss_mb, rss_mb)
                self.peak_cpu_percent = max(self.peak_cpu_percent, cpu)
            except psutil.Error as exc:
                LOGGER.debug("Resource sampling failed: %s", exc)
            time.sleep(self.interval_seconds)


@contextmanager
def timed_resource_block() -> Iterator[Dict[str, float]]:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    monitor = ResourceMonitor()
    start = time.perf_counter()
    monitor.start()
    metrics: Dict[str, float] = {}
    try:
        yield metrics
    finally:
        elapsed = time.perf_counter() - start
        resource_metrics = monitor.stop()
        metrics.update(resource_metrics)
        metrics["elapsed_seconds"] = elapsed
        metrics["gpu_peak_vram_mb"] = (
            float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
        )


def get_model_size_mb(model_name_or_path: str) -> float:
    path = Path(model_name_or_path)
    if not path.exists():
        return 0.0
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total / (1024**2)

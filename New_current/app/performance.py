"""Low-overhead pipeline timing and internally pollable progress state."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock
from time import perf_counter

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineTimings:
    """Collect and log high-resolution timings without changing API responses."""

    request_id: str
    started_at: float = field(default_factory=perf_counter)
    stages: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        started_at = perf_counter()
        try:
            yield
        finally:
            self.record(stage, (perf_counter() - started_at) * 1000)

    def record(self, stage: str, elapsed_ms: float) -> None:
        elapsed_ms = round(elapsed_ms, 3)
        self.stages[stage] = self.stages.get(stage, 0.0) + elapsed_ms
        logger.info(
            "%-24s %10.3f ms",
            stage,
            elapsed_ms,
            extra={
                "request_id": self.request_id,
                "pipeline_stage": stage,
                "stage_time_ms": elapsed_ms,
            },
        )

    def finish(self) -> float:
        elapsed_ms = round((perf_counter() - self.started_at) * 1000, 3)
        logger.info(
            "%-24s %10.3f ms",
            "Total Processing",
            elapsed_ms,
            extra={
                "request_id": self.request_id,
                "pipeline_stage": "Total Processing",
                "stage_time_ms": elapsed_ms,
                "pipeline_timings_ms": dict(self.stages),
            },
        )
        return elapsed_ms


@dataclass(frozen=True, slots=True)
class ProgressSnapshot:
    request_id: str
    stage: str
    percent: int
    complete: bool


class PipelineProgressStore:
    """Bounded process-local progress state for a future polling endpoint."""

    def __init__(self, max_entries: int = 512) -> None:
        self.max_entries = max_entries
        self._items: dict[str, ProgressSnapshot] = {}
        self._lock = RLock()

    def update(
        self,
        request_id: str,
        stage: str,
        percent: int,
        *,
        complete: bool = False,
    ) -> None:
        snapshot = ProgressSnapshot(request_id, stage, percent, complete)
        with self._lock:
            self._items[request_id] = snapshot
            while len(self._items) > self.max_entries:
                self._items.pop(next(iter(self._items)))

    def get(self, request_id: str) -> ProgressSnapshot | None:
        with self._lock:
            return self._items.get(request_id)

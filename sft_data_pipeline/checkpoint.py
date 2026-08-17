"""Atomic, crash-consistent checkpoint management."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class CheckpointError(RuntimeError):
    """Raised when checkpoint state is invalid or cannot be recovered."""


@dataclass
class CheckpointState:
    """Durable progress and optional in-flight transaction details."""

    last_processed_report_id: str | None = None
    last_processed_row_index: int = 0
    successful: int = 0
    failed: int = 0
    retries: int = 0
    samples_written: int = 0
    samples_per_report: int | None = None
    random_seed: int | None = None
    active_report_id: str | None = None
    active_row_index: int | None = None
    output_offset: int | None = None
    failed_offset: int | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CheckpointState":
        """Deserialize known checkpoint fields while rejecting bad types."""
        try:
            state = cls(
                last_processed_report_id=value.get(
                    "last_processed_report_id"
                ),
                last_processed_row_index=int(
                    value.get("last_processed_row_index", 0)
                ),
                successful=int(value.get("successful", 0)),
                failed=int(value.get("failed", 0)),
                retries=int(value.get("retries", 0)),
                samples_written=int(value.get("samples_written", 0)),
                samples_per_report=(
                    int(value["samples_per_report"])
                    if value.get("samples_per_report") is not None
                    else None
                ),
                random_seed=(
                    int(value["random_seed"])
                    if value.get("random_seed") is not None
                    else None
                ),
                active_report_id=value.get("active_report_id"),
                active_row_index=value.get("active_row_index"),
                output_offset=value.get("output_offset"),
                failed_offset=value.get("failed_offset"),
            )
        except (TypeError, ValueError) as exc:
            raise CheckpointError("Checkpoint contains invalid values.") from exc

        numeric_fields = (
            state.last_processed_row_index,
            state.successful,
            state.failed,
            state.retries,
            state.samples_written,
        )
        if any(item < 0 for item in numeric_fields):
            raise CheckpointError("Checkpoint counters cannot be negative.")
        if (
            state.samples_per_report is not None
            and state.samples_per_report <= 0
        ):
            raise CheckpointError(
                "Checkpoint samples_per_report must be positive."
            )
        return state


class CheckpointManager:
    """Persist progress atomically and recover interrupted file appends."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.state = self._load()

    def _load(self) -> CheckpointState:
        if not self.path.exists():
            return CheckpointState()
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(
                f"Cannot read checkpoint {self.path}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise CheckpointError("Checkpoint root must be a JSON object.")
        return CheckpointState.from_dict(value)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(asdict(self.state), handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except OSError as exc:
            raise CheckpointError(
                f"Cannot save checkpoint {self.path}: {exc}"
            ) from exc

    @staticmethod
    def _size(path: Path) -> int:
        return path.stat().st_size if path.exists() else 0

    @staticmethod
    def _truncate(path: Path, offset: int) -> None:
        if offset < 0:
            raise CheckpointError("A recovery offset cannot be negative.")
        current_size = CheckpointManager._size(path)
        if current_size < offset:
            raise CheckpointError(
                f"{path} is smaller than its checkpointed offset "
                f"({current_size} < {offset}); refusing unsafe recovery."
            )
        if current_size == offset:
            return
        with path.open("r+b") as handle:
            handle.truncate(offset)
            handle.flush()
            os.fsync(handle.fileno())

    def recover_incomplete(self, output: Path, failed: Path) -> bool:
        """Roll back appends from a report whose commit did not finish."""
        state = self.state
        if state.active_report_id is None:
            return False
        if state.output_offset is None or state.failed_offset is None:
            raise CheckpointError(
                "Active checkpoint is missing recovery file offsets."
            )

        self._truncate(output, int(state.output_offset))
        self._truncate(failed, int(state.failed_offset))
        state.active_report_id = None
        state.active_row_index = None
        state.output_offset = None
        state.failed_offset = None
        self._save()
        return True

    def bind_run_configuration(
        self,
        samples_per_report: int,
        random_seed: int,
    ) -> None:
        """Persist diversity settings and reject inconsistent resumes."""
        state = self.state
        if (
            state.samples_per_report is not None
            and state.samples_per_report != samples_per_report
        ):
            raise CheckpointError(
                "SAMPLES_PER_REPORT differs from the active checkpoint "
                f"({samples_per_report} != {state.samples_per_report})."
            )
        if state.random_seed is not None and state.random_seed != random_seed:
            raise CheckpointError(
                "RANDOM_SEED differs from the active checkpoint "
                f"({random_seed} != {state.random_seed})."
            )
        if (
            state.samples_per_report is None
            or state.random_seed is None
        ):
            state.samples_per_report = samples_per_report
            state.random_seed = random_seed
            self._save()

    def begin_report(
        self,
        report_id: str,
        row_index: int,
        output: Path,
        failed: Path,
    ) -> None:
        """Begin a crash-recoverable append transaction for one report."""
        if self.state.active_report_id is not None:
            raise CheckpointError("Another report is already active.")
        self.state.active_report_id = report_id
        self.state.active_row_index = row_index
        self.state.output_offset = self._size(output)
        self.state.failed_offset = self._size(failed)
        self._save()

    def complete_report(
        self,
        report_id: str,
        row_index: int,
        *,
        succeeded: bool,
        retries: int,
        samples_written: int = 0,
    ) -> None:
        """Commit one report after its result has been durably appended."""
        if (
            self.state.active_report_id != report_id
            or self.state.active_row_index != row_index
        ):
            raise CheckpointError("Completed report does not match active state.")

        self.state.last_processed_report_id = report_id
        self.state.last_processed_row_index = row_index
        self.state.successful += int(succeeded)
        self.state.failed += int(not succeeded)
        self.state.retries += retries
        self.state.samples_written += samples_written
        self.state.active_report_id = None
        self.state.active_row_index = None
        self.state.output_offset = None
        self.state.failed_offset = None
        self._save()

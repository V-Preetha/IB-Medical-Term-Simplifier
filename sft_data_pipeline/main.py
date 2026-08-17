"""Stream medical reports into a crash-resumable SFT JSONL dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from tqdm import tqdm

from checkpoint import CheckpointError, CheckpointManager
from config import INSTRUCTION_VARIANTS, Settings, load_settings
from generator import GenerationError, LLMGenerator
from validator import ValidationError, build_and_validate_sample

REQUIRED_COLUMNS = {
    "report_id",
    "specialty",
    "report_type",
    "difficulty",
    "report",
}
FAILED_COLUMNS = [
    "report_id",
    "specialty",
    "report_type",
    "difficulty",
    "report",
    "error",
    "attempts",
    "failed_at_utc",
]
LOGGER = logging.getLogger("medical_sft_pipeline")


def configure_logging(log_file: Path) -> None:
    """Configure file logging without exposing report contents."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
    )
    handler.formatter.converter = time.gmtime
    LOGGER.addHandler(handler)


def _set_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def inspect_input(path: Path) -> tuple[int, list[str]]:
    """Validate the CSV header and count rows without retaining them."""
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = REQUIRED_COLUMNS.difference(fieldnames)
        if missing:
            raise ValueError(
                f"Input CSV is missing required columns: {sorted(missing)}"
            )
        total = sum(1 for _ in reader)
    return total, fieldnames


def _append_jsonl(path: Path, sample: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        json.dump(
            sample,
            handle,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _append_failed(
    path: Path,
    row: dict[str, str],
    error: str,
    attempts: int,
) -> None:
    from datetime import datetime, timezone

    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    record = {column: row.get(column, "") for column in REQUIRED_COLUMNS}
    record.update(
        {
            "error": error.replace("\r", " ").replace("\n", " ")[:2000],
            "attempts": str(attempts),
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILED_COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow(record)
        handle.flush()
        os.fsync(handle.fileno())


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def select_instruction_variants(
    report_id: str,
    row_index: int,
    count: int,
    random_seed: int,
) -> list[str]:
    """Select reproducible shuffled variants without repeats per cycle."""
    seed_material = f"{random_seed}:{row_index}:{report_id}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest(), "big")
    generator = random.Random(seed)
    selected: list[str] = []

    while len(selected) < count:
        cycle = list(INSTRUCTION_VARIANTS)
        generator.shuffle(cycle)
        if selected and cycle[0] == selected[-1]:
            cycle[0], cycle[1] = cycle[1], cycle[0]
        selected.extend(cycle)
    return selected[:count]


def _verify_resume_row(
    row_index: int,
    report_id: str,
    checkpoint: CheckpointManager,
) -> None:
    state = checkpoint.state
    if row_index != state.last_processed_row_index:
        return
    if report_id != state.last_processed_report_id:
        raise CheckpointError(
            "reports.csv changed since the checkpoint: report_id at row "
            f"{row_index} is {report_id!r}, expected "
            f"{state.last_processed_report_id!r}."
        )


def _progress_postfix(
    checkpoint: CheckpointManager,
    started: float,
    session_processed: int,
    remaining: int,
    active_retries: int = 0,
) -> dict[str, str | int]:
    elapsed = time.monotonic() - started
    rate = session_processed / elapsed if elapsed > 0 else 0
    eta = remaining / rate if rate > 0 else None
    state = checkpoint.state
    return {
        "processed": state.last_processed_row_index,
        "successful": state.successful,
        "failed": state.failed,
        "samples": state.samples_written,
        "retries": state.retries + active_retries,
        "elapsed": _format_seconds(elapsed),
        "ETA": _format_seconds(eta),
    }


def run(settings: Settings) -> None:
    """Run the streaming generation pipeline."""
    _set_csv_field_limit()
    configure_logging(settings.log_file)
    total_rows, _ = inspect_input(settings.input_csv)
    checkpoint = CheckpointManager(settings.checkpoint_file)

    recovered = checkpoint.recover_incomplete(
        settings.output_jsonl,
        settings.failed_csv,
    )
    if recovered:
        LOGGER.warning("Recovered and rolled back an interrupted report append.")

    checkpoint.bind_run_configuration(
        settings.samples_per_report,
        settings.random_seed,
    )
    start_index = checkpoint.state.last_processed_row_index
    if start_index > total_rows:
        raise CheckpointError(
            "Checkpoint row exceeds the number of rows in reports.csv."
        )

    LOGGER.info(
        "Starting provider=%s model=%s total=%d resume_row=%d "
        "samples_per_report=%d self_check=%s",
        settings.provider.provider,
        settings.provider.model,
        total_rows,
        start_index,
        settings.samples_per_report,
        settings.enable_self_check,
    )
    started = time.monotonic()
    session_processed = 0

    with (
        settings.input_csv.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as input_handle,
        LLMGenerator(settings) as generator,
        tqdm(
            total=total_rows,
            initial=start_index,
            unit="report",
            desc="Generating SFT data",
            dynamic_ncols=True,
        ) as progress,
    ):
        reader = csv.DictReader(input_handle)
        verified_resume = start_index == 0

        for row_index, row in enumerate(reader, start=1):
            report_id = (row.get("report_id") or "").strip()
            if row_index <= start_index:
                if row_index == start_index:
                    _verify_resume_row(row_index, report_id, checkpoint)
                    verified_resume = True
                continue

            if not verified_resume:
                raise CheckpointError(
                    "Could not verify the checkpoint against reports.csv."
                )

            checkpoint.begin_report(
                report_id,
                row_index,
                settings.output_jsonl,
                settings.failed_csv,
            )
            active_retries = 0

            def on_retry(
                retry_number: int,
                error: Exception,
                delay: float,
            ) -> None:
                nonlocal active_retries
                active_retries = retry_number
                LOGGER.warning(
                    "Retry report_id=%s retry=%d delay=%.2fs error=%s",
                    report_id,
                    retry_number,
                    delay,
                    str(error).replace("\n", " ")[:500],
                )
                remaining = total_rows - row_index + 1
                progress.set_postfix(
                    _progress_postfix(
                        checkpoint,
                        started,
                        session_processed,
                        remaining,
                        active_retries,
                    ),
                    refresh=True,
                )

            succeeded = False
            retries = 0
            samples_written = 0
            try:
                report = (row.get("report") or "").strip()
                if not report_id:
                    raise ValidationError("'report_id' is empty.")
                if not report:
                    raise ValidationError("'report' is empty.")
                assistant_content, retries = generator.generate(
                    report,
                    on_retry=on_retry,
                )
                instructions = select_instruction_variants(
                    report_id,
                    row_index,
                    settings.samples_per_report,
                    settings.random_seed,
                )
                for instruction in instructions:
                    sample = build_and_validate_sample(
                        report,
                        assistant_content,
                        instruction,
                    )
                    _append_jsonl(settings.output_jsonl, sample)
                    samples_written += 1
                succeeded = True
            except GenerationError as exc:
                retries = exc.retries
                _append_failed(
                    settings.failed_csv,
                    row,
                    str(exc),
                    retries + 1,
                )
                LOGGER.error(
                    "Permanently failed report_id=%s attempts=%d error=%s",
                    report_id,
                    retries + 1,
                    str(exc).replace("\n", " ")[:500],
                )
            except (ValidationError, ValueError) as exc:
                _append_failed(
                    settings.failed_csv,
                    row,
                    str(exc),
                    0,
                )
                LOGGER.error(
                    "Rejected input report_id=%s error=%s",
                    report_id or "<empty>",
                    str(exc).replace("\n", " ")[:500],
                )

            checkpoint.complete_report(
                report_id,
                row_index,
                succeeded=succeeded,
                retries=retries,
                samples_written=samples_written,
            )
            session_processed += 1
            progress.update(1)
            postfix = _progress_postfix(
                checkpoint,
                started,
                session_processed,
                total_rows - row_index,
            )
            progress.set_postfix(
                postfix,
                refresh=True,
            )
            LOGGER.info(
                "Processed report_id=%s status=%s processed=%d "
                "successful=%d failed=%d samples=%d retries=%d "
                "elapsed=%s eta=%s",
                report_id or "<empty>",
                "successful" if succeeded else "failed",
                checkpoint.state.last_processed_row_index,
                checkpoint.state.successful,
                checkpoint.state.failed,
                checkpoint.state.samples_written,
                checkpoint.state.retries,
                postfix["elapsed"],
                postfix["ETA"],
            )

    elapsed = time.monotonic() - started
    LOGGER.info(
        "Finished processed=%d successful=%d failed=%d samples=%d "
        "retries=%d elapsed_seconds=%.2f eta_seconds=0",
        checkpoint.state.last_processed_row_index,
        checkpoint.state.successful,
        checkpoint.state.failed,
        checkpoint.state.samples_written,
        checkpoint.state.retries,
        elapsed,
    )


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line overrides while retaining environment defaults."""
    parser = argparse.ArgumentParser(
        description="Generate validated medical-report SFT JSONL data."
    )
    parser.add_argument(
        "--samples-per-report",
        type=_positive_integer,
        default=None,
        metavar="N",
        help="instruction variants per report (default: 5)",
    )
    parser.add_argument(
        "--self-check",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable the post-generation medical fact audit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point with concise, actionable failures."""
    try:
        args = parse_args(argv)
        settings = load_settings()
        if args.samples_per_report is not None:
            settings = replace(
                settings,
                samples_per_report=args.samples_per_report,
            )
        if args.self_check is not None:
            settings = replace(
                settings,
                enable_self_check=args.self_check,
            )
        run(settings)
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted; the active row will be recovered on resume.")
        print("\nInterrupted. Re-run the command to resume safely.", file=sys.stderr)
        return 130
    except Exception as exc:
        LOGGER.exception("Pipeline stopped: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

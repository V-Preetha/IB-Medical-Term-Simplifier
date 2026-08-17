"""Shared logging, reproducibility, hardware, and metrics utilities."""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


LOGGER = logging.getLogger("medical_term_sft")


def configure_logging(verbose: bool = False) -> None:
    """Configure a consistent process-wide logger."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def set_reproducibility(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch and enable deterministic behavior."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def hardware_summary() -> dict[str, Any]:
    """Return hardware details and log the selected backend."""

    if torch.cuda.is_available():
        try:
            probe = torch.ones(1, device="cuda")
            torch.cuda.synchronize()
            del probe
        except RuntimeError as error:
            raise RuntimeError(
                "CUDA is visible, but this PyTorch build cannot execute a "
                "kernel on the GPU. Install a wheel containing the GPU's "
                f"compute capability. PyTorch={torch.__version__}, "
                f"supported_arches={torch.cuda.get_arch_list()}"
            ) from error
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "vram_gib": round(
                        properties.total_memory / (1024**3), 2
                    ),
                }
            )
        summary = {
            "backend": "cuda",
            "device_count": len(devices),
            "devices": devices,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    elif torch.backends.mps.is_available():
        summary = {
            "backend": "mps",
            "device_count": 1,
            "devices": [{"index": 0, "name": "Apple Metal", "vram_gib": None}],
            "bf16_supported": False,
        }
    else:
        summary = {
            "backend": "cpu",
            "device_count": 1,
            "devices": [{"index": 0, "name": "CPU", "vram_gib": None}],
            "bf16_supported": False,
        }
    LOGGER.info("Hardware: %s", json.dumps(summary, ensure_ascii=False))
    return summary


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write JSON so interruption cannot leave a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def count_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return trainable and total parameter counts."""

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    percentage = 100.0 * trainable / max(total, 1)
    LOGGER.info(
        "Parameters: %s trainable / %s total (%.4f%%)",
        f"{trainable:,}",
        f"{total:,}",
        percentage,
    )
    return trainable, total


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    """Convert a nested dataclass to JSON-compatible primitives."""

    if not is_dataclass(value):
        raise TypeError("Expected a dataclass instance.")

    def normalize(item: Any) -> Any:
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, tuple):
            return [normalize(element) for element in item]
        if isinstance(item, dict):
            return {key: normalize(element) for key, element in item.items()}
        if isinstance(item, list):
            return [normalize(element) for element in item]
        return item

    return normalize(asdict(value))


class MetricsCallback(TrainerCallback):
    """Persist Trainer logs to CSV and continuously refresh a loss plot."""

    fieldnames = ("step", "epoch", "training_loss", "validation_loss")

    def __init__(self, csv_path: Path, plot_path: Path) -> None:
        self.csv_path = csv_path
        self.plot_path = plot_path
        self.rows: list[dict[str, float | int | None]] = []
        self._last_training_loss: float | None = None
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if csv_path.is_file():
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    parsed = {
                        "step": int(row["step"]),
                        "epoch": float(row["epoch"]),
                        "training_loss": (
                            float(row["training_loss"])
                            if row["training_loss"]
                            else None
                        ),
                        "validation_loss": (
                            float(row["validation_loss"])
                            if row["validation_loss"]
                            else None
                        ),
                    }
                    self.rows.append(parsed)
                    if parsed["training_loss"] is not None:
                        self._last_training_loss = float(
                            parsed["training_loss"]
                        )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **_: Any,
    ) -> None:
        """Capture train and validation loss logs."""

        del args, control
        if not state.is_world_process_zero:
            return
        if not logs:
            return
        if "loss" in logs:
            self._last_training_loss = float(logs["loss"])
        if "loss" not in logs and "eval_loss" not in logs:
            return
        self.rows.append(
            {
                "step": state.global_step,
                "epoch": float(state.epoch or 0.0),
                "training_loss": (
                    float(logs["loss"])
                    if "loss" in logs
                    else self._last_training_loss
                ),
                "validation_loss": (
                    float(logs["eval_loss"])
                    if "eval_loss" in logs
                    else None
                ),
            }
        )
        self._write_csv()
        self._plot()

    def _write_csv(self) -> None:
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

    def _plot(self) -> None:
        train_rows = [
            row for row in self.rows if row["training_loss"] is not None
        ]
        validation_rows = [
            row for row in self.rows if row["validation_loss"] is not None
        ]
        if not train_rows:
            return
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.plot(
            [float(row["epoch"] or 0.0) for row in train_rows],
            [float(row["training_loss"]) for row in train_rows],
            label="Training loss",
            alpha=0.8,
        )
        if validation_rows:
            axis.plot(
                [float(row["epoch"] or 0.0) for row in validation_rows],
                [float(row["validation_loss"]) for row in validation_rows],
                marker="o",
                label="Validation loss",
            )
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.set_title("SFT training and validation loss")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(self.plot_path, dpi=160)
        plt.close(figure)

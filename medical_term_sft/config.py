"""Typed configuration loading for the medical QLoRA project."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
T = TypeVar("T")


@dataclass(frozen=True)
class ModelConfig:
    """Base model and model-loading settings."""

    name: str = "Qwen/Qwen3-4B-Instruct-2507"
    trust_remote_code: bool = False
    use_flash_attention: bool = True
    use_4bit: bool = True


@dataclass(frozen=True)
class DataConfig:
    """Dataset and preprocessing settings."""

    path: Path = Path("../sft_data_pipeline/medical_simplifier_synthetic_5000.jsonl")
    split_indices_path: Path = Path("outputs/split_indices.json")
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    max_sequence_length: int = 4096
    num_proc: int = 1
    inject_report_into_user: bool = True


@dataclass(frozen=True)
class LoraSettings:
    """LoRA adapter settings."""

    rank: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_attention_projections: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization, checkpointing, and logging settings."""

    output_dir: Path = Path("outputs")
    checkpoint_dir: Path = Path("checkpoints")
    logging_dir: Path = Path("outputs/tensorboard")
    num_train_epochs: float = 3.0
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    optimizer: str = "adamw_torch"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_total_limit: int = 3
    early_stopping_patience: int = 3
    dataloader_num_workers: int = 0
    estimated_tokens_per_second_per_gpu: float = 1800.0
    seed: int = 42
    full_determinism: bool = True
    auto_resume: bool = True


@dataclass(frozen=True)
class GenerationConfig:
    """Generation and post-training evaluation settings."""

    max_new_tokens: int = 2048
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    prediction_samples: int = 25


@dataclass(frozen=True)
class ProjectConfig:
    """Complete application configuration."""

    model: ModelConfig
    data: DataConfig
    lora: LoraSettings
    training: TrainingConfig
    generation: GenerationConfig
    config_path: Path


def _known_values(cls: type[T], values: dict[str, Any]) -> dict[str, Any]:
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(
            f"Unknown keys in {cls.__name__}: {', '.join(unknown)}"
        )
    return values


def _resolve_path(value: str | Path, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    return path


def load_config(config_path: str | Path) -> ProjectConfig:
    """Load and validate a YAML configuration file."""

    resolved_config = Path(config_path).expanduser().resolve()
    with resolved_config.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("The YAML root must be a mapping.")

    config_dir = resolved_config.parent
    model_raw = _known_values(ModelConfig, dict(raw.get("model", {})))
    data_raw = _known_values(DataConfig, dict(raw.get("data", {})))
    lora_raw = _known_values(LoraSettings, dict(raw.get("lora", {})))
    training_raw = _known_values(
        TrainingConfig, dict(raw.get("training", {}))
    )
    generation_raw = _known_values(
        GenerationConfig, dict(raw.get("generation", {}))
    )

    for key in ("path", "split_indices_path"):
        if key in data_raw:
            data_raw[key] = _resolve_path(data_raw[key], config_dir)
    for key in ("output_dir", "checkpoint_dir", "logging_dir"):
        if key in training_raw:
            training_raw[key] = _resolve_path(training_raw[key], config_dir)
    if "target_attention_projections" in lora_raw:
        lora_raw["target_attention_projections"] = tuple(
            lora_raw["target_attention_projections"]
        )

    config = ProjectConfig(
        model=ModelConfig(**model_raw),
        data=DataConfig(**data_raw),
        lora=LoraSettings(**lora_raw),
        training=TrainingConfig(**training_raw),
        generation=GenerationConfig(**generation_raw),
        config_path=resolved_config,
    )
    _validate_config(config)
    return config


def _validate_config(config: ProjectConfig) -> None:
    ratios = (
        config.data.train_ratio,
        config.data.validation_ratio,
        config.data.test_ratio,
    )
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("All data split ratios must be positive.")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Data split ratios must sum to 1.0.")
    if config.data.max_sequence_length < 128:
        raise ValueError("max_sequence_length must be at least 128.")
    if config.lora.rank <= 0 or config.lora.alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive.")
    if config.training.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "qwen3.yaml"

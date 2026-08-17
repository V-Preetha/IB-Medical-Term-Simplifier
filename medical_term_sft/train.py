"""Command-line entry point for Qwen3 medical QLoRA training."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from config import DEFAULT_CONFIG_PATH, ProjectConfig, load_config
from dataset import prepare_datasets
from evaluate import evaluate_after_training
from trainer import (
    create_trainer,
    estimate_training_time,
    find_resume_checkpoint,
    load_qlora_model,
    load_tokenizer,
)
from utils import (
    atomic_write_json,
    configure_logging,
    dataclass_to_dict,
    hardware_summary,
    set_reproducibility,
)


LOGGER = logging.getLogger("medical_term_sft.train")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3 for medical report simplification."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML configuration path.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Optional JSONL path override; the file is never modified.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start a new run instead of resuming the latest checkpoint.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def apply_overrides(config: ProjectConfig, args: argparse.Namespace) -> ProjectConfig:
    """Apply explicit CLI overrides without mutating typed config objects."""

    data = config.data
    training = config.training
    if args.dataset:
        data = replace(data, path=args.dataset.expanduser().resolve())
    if args.no_resume:
        training = replace(training, auto_resume=False)
    return replace(config, data=data, training=training)


def main() -> None:
    """Run end-to-end training, selection, and test evaluation."""

    args = parse_args()
    configure_logging(args.verbose)
    config = apply_overrides(load_config(args.config), args)
    set_reproducibility(
        config.training.seed, config.training.full_determinism
    )
    hardware_summary()
    config.training.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        config.training.output_dir / "resolved_config.json",
        dataclass_to_dict(config),
    )

    tokenizer = load_tokenizer(config)
    datasets = prepare_datasets(config.data, tokenizer, config.training.seed)
    estimate_training_time(datasets, config)
    model = load_qlora_model(config)
    trainer = create_trainer(model, tokenizer, datasets, config)
    checkpoint = find_resume_checkpoint(config)
    result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_state()
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)

    adapter_path = config.training.output_dir / "best_adapter"
    trainer.model.config.use_cache = True
    trainer.save_model(str(adapter_path))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(adapter_path)
        atomic_write_json(
            config.training.checkpoint_dir / "best.json",
            {
                "checkpoint": trainer.state.best_model_checkpoint,
                "validation_loss": trainer.state.best_metric,
                "exported_adapter": str(adapter_path),
            },
        )
        LOGGER.info("Saved best LoRA adapter to %s", adapter_path)
    evaluate_after_training(trainer, tokenizer, datasets, config)
    LOGGER.info("Training and evaluation completed successfully.")


if __name__ == "__main__":
    main()

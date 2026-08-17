"""QLoRA model construction and TRL trainer orchestration."""

from __future__ import annotations

import importlib.util
import logging
import math
import os
from pathlib import Path
from typing import Any

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import TrainingArguments
from trl import SFTConfig, SFTTrainer

from config import ProjectConfig
from dataset import DatasetBundle
from utils import MetricsCallback, atomic_write_json, count_trainable_parameters


LOGGER = logging.getLogger("medical_term_sft.trainer")


class AssistantOnlyDataCollator(DataCollatorForSeq2Seq):
    """Dynamically pad tokenized rows and discard estimation metadata."""

    def __call__(
        self,
        features: list[dict[str, Any]],
        return_tensors: str | None = None,
    ) -> dict[str, Any]:
        cleaned = [
            {key: value for key, value in feature.items() if key != "length"}
            for feature in features
        ]
        return super().__call__(cleaned, return_tensors=return_tensors)


class CheckpointPointerCallback(TrainerCallback):
    """Write small, portable pointers to latest and best checkpoints."""

    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = checkpoint_dir

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **_: Any,
    ) -> None:
        del args, control
        if not state.is_world_process_zero:
            return
        latest = self.checkpoint_dir / f"checkpoint-{state.global_step}"
        atomic_write_json(
            self.checkpoint_dir / "latest.json",
            {"checkpoint": str(latest), "global_step": state.global_step},
        )
        if state.best_model_checkpoint:
            atomic_write_json(
                self.checkpoint_dir / "best.json",
                {
                    "checkpoint": state.best_model_checkpoint,
                    "metric": state.best_metric,
                },
            )


def load_tokenizer(config: ProjectConfig) -> PreTrainedTokenizerBase:
    """Load and normalize the configured tokenizer."""

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        trust_remote_code=config.model.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _attention_implementation(config: ProjectConfig) -> str:
    flash_available = (
        config.model.use_flash_attention
        and torch.cuda.is_available()
        and importlib.util.find_spec("flash_attn") is not None
    )
    implementation = "flash_attention_2" if flash_available else "sdpa"
    LOGGER.info("Attention implementation: %s", implementation)
    return implementation


def _quantization_config(config: ProjectConfig, dtype: torch.dtype) -> Any:
    if not config.model.use_4bit:
        return None
    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA training requires a CUDA GPU in this project. Hardware was "
            "detected successfully, but 4-bit bitsandbytes training is not "
            "enabled for CPU/MPS. Set model.use_4bit=false only for a deliberate "
            "full-precision development run."
        )
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )


def load_qlora_model(config: ProjectConfig) -> PreTrainedModel:
    """Load the base model in NF4 and attach trainable LoRA adapters."""

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    quantization = _quantization_config(config, dtype)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_map = {"": local_rank} if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        trust_remote_code=config.model.trust_remote_code,
        quantization_config=quantization,
        dtype=dtype if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        attn_implementation=_attention_implementation(config),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if config.model.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.training.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    targets = detect_attention_projections(
        model, config.lora.target_attention_projections
    )
    adapter_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, adapter_config)
    count_trainable_parameters(model)
    return model


def detect_attention_projections(
    model: torch.nn.Module, expected: tuple[str, ...]
) -> list[str]:
    """Detect and verify every requested Qwen attention projection."""

    module_names = [name for name, _ in model.named_modules()]
    detected = [
        suffix
        for suffix in expected
        if any(name.endswith(f".{suffix}") for name in module_names)
    ]
    missing = sorted(set(expected) - set(detected))
    if missing:
        raise RuntimeError(
            "Model is missing configured attention projections: "
            + ", ".join(missing)
        )
    counts = {
        suffix: sum(name.endswith(f".{suffix}") for name in module_names)
        for suffix in detected
    }
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"Uneven attention projection coverage: {counts}")
    LOGGER.info("LoRA attention targets: %s", counts)
    return detected


def estimate_training_time(
    datasets: DatasetBundle, config: ProjectConfig
) -> float:
    """Estimate wall-clock hours from tokens and a configurable throughput."""

    tokens = sum(int(value) for value in datasets.train["length"])
    world_size = max(int(os.environ.get("WORLD_SIZE", "1")), 1)
    throughput = (
        config.training.estimated_tokens_per_second_per_gpu * world_size
    )
    hours = (
        tokens * config.training.num_train_epochs / max(throughput, 1.0) / 3600
    )
    LOGGER.info(
        "Estimated training time: %.2f hours (%s tokens/epoch, %.0f "
        "configured tokens/s across %d process(es)).",
        hours,
        f"{tokens:,}",
        throughput,
        world_size,
    )
    return hours


def create_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    datasets: DatasetBundle,
    config: ProjectConfig,
) -> SFTTrainer:
    """Create a current TRL SFTTrainer for pre-tokenized assistant-only data."""

    training = config.training
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training.output_dir.mkdir(parents=True, exist_ok=True)
    training.logging_dir.mkdir(parents=True, exist_ok=True)
    args = SFTConfig(
        output_dir=str(training.checkpoint_dir),
        num_train_epochs=training.num_train_epochs,
        learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        warmup_ratio=training.warmup_ratio,
        lr_scheduler_type=training.lr_scheduler_type,
        optim=training.optimizer,
        per_device_train_batch_size=training.per_device_train_batch_size,
        per_device_eval_batch_size=training.per_device_eval_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        max_grad_norm=training.max_grad_norm,
        gradient_checkpointing=training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=training.logging_steps,
        logging_first_step=True,
        save_total_limit=training.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        logging_dir=str(training.logging_dir),
        seed=training.seed,
        data_seed=training.seed,
        full_determinism=training.full_determinism,
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        tf32=torch.cuda.is_available(),
        max_length=config.data.max_sequence_length,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        dataloader_num_workers=training.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        ddp_find_unused_parameters=False,
        include_num_input_tokens_seen=True,
    )
    collator = AssistantOnlyDataCollator(
        tokenizer=tokenizer,
        model=None,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
        return_tensors="pt",
    )

    callbacks: list[TrainerCallback] = [
        EarlyStoppingCallback(
            early_stopping_patience=training.early_stopping_patience
        ),
        MetricsCallback(
            training.output_dir / "training_metrics.csv",
            training.output_dir / "loss_curves.png",
        ),
        CheckpointPointerCallback(training.checkpoint_dir),
    ]
    return SFTTrainer(
        model=model,
        args=args,
        train_dataset=datasets.train,
        eval_dataset=datasets.validation,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=callbacks,
    )


def find_resume_checkpoint(config: ProjectConfig) -> str | None:
    """Find the latest valid Trainer checkpoint when auto-resume is enabled."""

    if not config.training.auto_resume:
        return None
    directory = config.training.checkpoint_dir
    if not directory.is_dir():
        return None
    checkpoint = get_last_checkpoint(str(directory))
    if checkpoint:
        LOGGER.info("Automatically resuming from %s", checkpoint)
    return checkpoint

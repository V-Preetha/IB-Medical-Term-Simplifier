"""Dataset validation, splitting, chat rendering, and tokenization."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from config import DataConfig
from utils import atomic_write_json


LOGGER = logging.getLogger("medical_term_sft.dataset")
REQUIRED_OUTPUT_FIELDS = {"report", "summary", "simplification", "entities"}


@dataclass(frozen=True)
class DatasetBundle:
    """Raw and tokenized deterministic splits."""

    raw_train: Dataset
    raw_validation: Dataset
    raw_test: Dataset
    train: Dataset
    validation: Dataset
    test: Dataset


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_row(example: Mapping[str, Any], index: int) -> None:
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError(f"Row {index}: messages must contain at least 3 turns.")
    roles = [message.get("role") for message in messages]
    if roles[-1] != "assistant":
        raise ValueError(f"Row {index}: the final turn must be assistant.")
    assistant = messages[-1].get("content")
    if isinstance(assistant, str):
        try:
            assistant = json.loads(assistant)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Row {index}: assistant content is not valid JSON."
            ) from error
    if not isinstance(assistant, dict):
        raise ValueError(f"Row {index}: assistant content must be an object.")
    missing = REQUIRED_OUTPUT_FIELDS - set(assistant)
    if missing:
        raise ValueError(
            f"Row {index}: assistant content is missing {sorted(missing)}."
        )


def load_raw_dataset(path: Path) -> Dataset:
    """Load JSONL with Hugging Face Datasets without modifying the file."""

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    dataset = load_dataset("json", data_files=str(path), split="train")
    for index, example in enumerate(dataset):
        _validate_row(example, index)
    LOGGER.info(
        "Validated %s source examples from %s", f"{len(dataset):,}", path
    )
    return dataset


def _create_split_indices(
    size: int, config: DataConfig, seed: int
) -> dict[str, list[int]]:
    generator = np.random.default_rng(seed)
    indices = generator.permutation(size).tolist()
    train_end = int(size * config.train_ratio)
    validation_end = train_end + int(size * config.validation_ratio)
    return {
        "train": indices[:train_end],
        "validation": indices[train_end:validation_end],
        "test": indices[validation_end:],
    }


def load_or_create_split_indices(
    dataset_path: Path,
    size: int,
    config: DataConfig,
    seed: int,
) -> dict[str, list[int]]:
    """Reuse saved split indices, rejecting stale indices safely."""

    fingerprint = _file_fingerprint(dataset_path)
    path = config.split_indices_path
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("fingerprint") != fingerprint or payload.get("size") != size:
            raise RuntimeError(
                "Saved split indices belong to a different dataset. Move or "
                f"delete {path} explicitly before using the changed dataset."
            )
        if payload.get("seed") != seed:
            raise RuntimeError(
                f"Saved splits use seed {payload.get('seed')}, not {seed}."
            )
        return {
            name: [int(index) for index in payload[name]]
            for name in ("train", "validation", "test")
        }

    splits = _create_split_indices(size, config, seed)
    atomic_write_json(
        path,
        {
            "dataset": str(dataset_path),
            "fingerprint": fingerprint,
            "size": size,
            "seed": seed,
            **splits,
        },
    )
    LOGGER.info("Saved deterministic split indices to %s", path)
    return splits


def assistant_object(example: Mapping[str, Any]) -> dict[str, Any]:
    """Return the structured reference assistant object."""

    content = example["messages"][-1]["content"]
    if isinstance(content, str):
        content = json.loads(content)
    if not isinstance(content, dict):
        raise TypeError("Assistant content must be a JSON object.")
    return content


def build_prompt_messages(
    example: Mapping[str, Any], inject_report: bool = True
) -> list[dict[str, str]]:
    """Build inference-equivalent messages while leaving source rows untouched."""

    source_messages = example["messages"][:-1]
    messages = [
        {"role": str(item["role"]), "content": str(item["content"])}
        for item in source_messages
    ]
    if inject_report:
        report = str(assistant_object(example)["report"]).strip()
        if not report:
            raise ValueError("Reference report cannot be empty.")
        user_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if messages[index]["role"] == "user"
            ),
            None,
        )
        if user_index is None:
            raise ValueError("Each row must contain a user turn.")
        instruction = messages[user_index]["content"].rstrip()
        messages[user_index] = {
            "role": "user",
            "content": f"{instruction}\n\nMedical report:\n{report}",
        }
    return messages


def build_training_messages(
    example: Mapping[str, Any], inject_report: bool = True
) -> list[dict[str, str]]:
    """Serialize structured assistant content as canonical valid JSON."""

    messages = build_prompt_messages(example, inject_report)
    response = json.dumps(
        assistant_object(example),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    messages.append({"role": "assistant", "content": response})
    return messages


def render_prompt(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[Mapping[str, str]],
) -> str:
    """Render Qwen's native prompt with thinking disabled when supported."""

    return tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _tokenize_example(
    example: Mapping[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    config: DataConfig,
) -> dict[str, Any]:
    prompt_messages = build_prompt_messages(
        example, config.inject_report_into_user
    )
    full_messages = build_training_messages(
        example, config.inject_report_into_user
    )
    prompt_text = render_prompt(tokenizer, prompt_messages)
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if not full_text.startswith(prompt_text):
        raise ValueError(
            "Chat template full conversation is not prefixed by its prompt; "
            "assistant-only masking cannot be guaranteed."
        )

    prompt_ids = tokenizer(
        prompt_text, add_special_tokens=False, truncation=False
    )["input_ids"]
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=config.max_sequence_length,
    )
    input_ids = list(encoded["input_ids"])
    assistant_start = min(len(prompt_ids), len(input_ids))
    labels = [-100] * assistant_start + input_ids[assistant_start:]
    if all(label == -100 for label in labels):
        raise ValueError(
            "The prompt consumes max_sequence_length and leaves no assistant "
            "tokens. Increase the limit or shorten the source report."
        )
    return {
        "input_ids": input_ids,
        "attention_mask": list(encoded["attention_mask"]),
        "labels": labels,
        "length": len(input_ids),
    }


def prepare_datasets(
    config: DataConfig,
    tokenizer: PreTrainedTokenizerBase,
    seed: int,
) -> DatasetBundle:
    """Load, split, and pre-tokenize all deterministic dataset splits."""

    raw = load_raw_dataset(config.path)
    indices = load_or_create_split_indices(config.path, len(raw), config, seed)
    raw_splits = {
        name: raw.select(split_indices)
        for name, split_indices in indices.items()
    }

    def tokenize(batch: Mapping[str, Any]) -> dict[str, Any]:
        return _tokenize_example(batch, tokenizer, config)

    tokenized: dict[str, Dataset] = {}
    for name, split in raw_splits.items():
        LOGGER.info(
            "Tokenizing %s split (%s examples)", name, f"{len(split):,}"
        )
        tokenized[name] = split.map(
            tokenize,
            remove_columns=split.column_names,
            num_proc=config.num_proc,
            desc=f"Tokenizing {name}",
        )
    return DatasetBundle(
        raw_train=raw_splits["train"],
        raw_validation=raw_splits["validation"],
        raw_test=raw_splits["test"],
        train=tokenized["train"],
        validation=tokenized["validation"],
        test=tokenized["test"],
    )

"""Loss, structured-generation, and medical-fidelity evaluation."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import SFTTrainer

from config import ProjectConfig
from dataset import DatasetBundle, assistant_object, build_prompt_messages
from utils import atomic_write_json


LOGGER = logging.getLogger("medical_term_sft.evaluate")
DOSAGE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mcg|mg|g|kg|mL|ml|L|units?|IU|mEq)"
    r"(?:/[A-Za-z]+)?\b",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(
    r"(?<![\w.])[<>≤≥±]?\s*-?\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?"
    r"\s*(?:%|mmHg|bpm|°[CF])?",
    re.IGNORECASE,
)
REQUIRED_PATHS = (
    ("report",),
    ("summary",),
    ("simplification",),
    ("simplification", "clinical"),
    ("simplification", "general"),
    ("simplification", "child"),
    ("entities",),
)


@dataclass(frozen=True)
class ParseResult:
    """Strict JSON parsing result for one model completion."""

    value: dict[str, Any] | None
    error: str | None


def parse_generated_json(text: str) -> ParseResult:
    """Parse a completion strictly; markdown fences count as invalid JSON."""

    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as error:
        return ParseResult(
            None,
            f"{error.msg} at line {error.lineno}, column {error.colno}",
        )
    if not isinstance(value, dict):
        return ParseResult(None, "Top-level JSON value is not an object.")
    return ParseResult(value, None)


def missing_fields(value: Mapping[str, Any] | None) -> list[str]:
    """Return absent, null, or empty required output paths."""

    if value is None:
        return [".".join(path) for path in REQUIRED_PATHS]
    missing: list[str] = []
    for path in REQUIRED_PATHS:
        current: Any = value
        for part in path:
            if not isinstance(current, Mapping) or part not in current:
                current = None
                break
            current = current[part]
        if current is None or current == "" or current == []:
            missing.append(".".join(path))
    return missing


def _flatten_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return " ".join(_flatten_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_flatten_text(item) for item in value)
    return str(value)


def _normalized_set(values: Iterable[str]) -> set[str]:
    return {" ".join(value.lower().split()) for value in values if value.strip()}


def _recall(reference: set[str], generated_text: str) -> float | None:
    if not reference:
        return None
    normalized_output = " ".join(generated_text.lower().split())
    preserved = sum(item in normalized_output for item in reference)
    return preserved / len(reference)


def medical_fidelity(
    reference: Mapping[str, Any], generated: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Compute transparent lexical preservation checks for critical facts.

    Diagnoses and medications come from annotated entities. Dosages and all
    numerical values are extracted from the source report. This is an automatic
    regression signal, not a replacement for clinician review.
    """

    entities = reference.get("entities", [])
    diagnoses = _normalized_set(
        str(entity.get("term", ""))
        for entity in entities
        if entity.get("type") == "Disease"
    )
    medications = _normalized_set(
        str(entity.get("term", ""))
        for entity in entities
        if entity.get("type") == "Medication"
    )
    report = str(reference.get("report", ""))
    dosages = _normalized_set(DOSAGE_PATTERN.findall(report))
    numbers = _normalized_set(
        match.group(0) for match in NUMBER_PATTERN.finditer(report)
    )
    generated_text = _flatten_text(generated) if generated is not None else ""
    category_scores = {
        "diagnosis_recall": _recall(diagnoses, generated_text),
        "medication_recall": _recall(medications, generated_text),
        "dosage_recall": _recall(dosages, generated_text),
        "numerical_value_recall": _recall(numbers, generated_text),
    }
    applicable = [score for score in category_scores.values() if score is not None]
    score = 100.0 * sum(applicable) / len(applicable) if applicable else 100.0
    return {
        "medical_fidelity_score": round(score, 2),
        **{
            name: None if value is None else round(100.0 * value, 2)
            for name, value in category_scores.items()
        },
        "reference_fact_counts": {
            "diagnoses": len(diagnoses),
            "medications": len(medications),
            "dosages": len(dosages),
            "numerical_values": len(numbers),
        },
    }


def _generate_one(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    config: ProjectConfig,
) -> tuple[str, int]:
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    )
    model_device = next(model.parameters()).device
    inputs = {name: tensor.to(model_device) for name, tensor in inputs.items()}
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": config.generation.max_new_tokens,
        "do_sample": config.generation.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if config.generation.do_sample:
        generation_kwargs.update(
            temperature=config.generation.temperature,
            top_p=config.generation.top_p,
        )
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    prompt_length = inputs["input_ids"].shape[-1]
    completion_ids = output_ids[0, prompt_length:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True), int(
        completion_ids.numel()
    )


def evaluate_after_training(
    trainer: SFTTrainer,
    tokenizer: PreTrainedTokenizerBase,
    datasets: DatasetBundle,
    config: ProjectConfig,
) -> dict[str, Any]:
    """Compute final losses and structured metrics on sampled test reports."""

    validation_metrics = trainer.evaluate(
        datasets.validation, metric_key_prefix="validation"
    )
    test_metrics = trainer.evaluate(datasets.test, metric_key_prefix="test")
    if not trainer.is_world_process_zero():
        return {
            "validation_loss": validation_metrics.get("validation_loss"),
            "test_loss": test_metrics.get("test_loss"),
        }
    sample_count = min(config.generation.prediction_samples, len(datasets.raw_test))
    generator = np.random.default_rng(config.training.seed)
    selected = generator.choice(
        len(datasets.raw_test), size=sample_count, replace=False
    ).tolist()
    predictions: list[dict[str, Any]] = []
    model = trainer.model
    model.eval()
    for test_index in tqdm(selected, desc="Generating test predictions"):
        example = datasets.raw_test[int(test_index)]
        reference = assistant_object(example)
        messages = build_prompt_messages(
            example, config.data.inject_report_into_user
        )
        raw, token_count = _generate_one(model, tokenizer, messages, config)
        parsed = parse_generated_json(raw)
        absent = missing_fields(parsed.value)
        fidelity = medical_fidelity(reference, parsed.value)
        entity_count = 0
        if parsed.value is not None and isinstance(parsed.value.get("entities"), list):
            entity_count = len(parsed.value["entities"])
        predictions.append(
            {
                "test_index": int(test_index),
                "messages": messages,
                "reference": reference,
                "raw_prediction": raw,
                "parsed_prediction": parsed.value,
                "json_valid": parsed.error is None,
                "parse_error": parsed.error,
                "missing_fields": absent,
                "generation_tokens": token_count,
                "entity_count": entity_count,
                "fidelity": fidelity,
            }
        )

    valid_count = sum(item["json_valid"] for item in predictions)
    aggregate = {
        "validation_loss": validation_metrics.get("validation_loss"),
        "test_loss": test_metrics.get("test_loss"),
        "prediction_count": len(predictions),
        "json_validity_percent": round(
            100.0 * valid_count / max(len(predictions), 1), 2
        ),
        "average_generation_length_tokens": round(
            sum(item["generation_tokens"] for item in predictions)
            / max(len(predictions), 1),
            2,
        ),
        "average_entity_count": round(
            sum(item["entity_count"] for item in predictions)
            / max(len(predictions), 1),
            2,
        ),
        "outputs_with_missing_fields": sum(
            bool(item["missing_fields"]) for item in predictions
        ),
        "medical_fidelity_score": round(
            sum(item["fidelity"]["medical_fidelity_score"] for item in predictions)
            / max(len(predictions), 1),
            2,
        ),
    }
    config.training.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(config.training.output_dir / "predictions.json", predictions)
    atomic_write_json(
        config.training.output_dir / "evaluation_metrics.json", aggregate
    )
    LOGGER.info("Final evaluation: %s", json.dumps(aggregate))
    return aggregate

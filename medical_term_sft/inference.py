"""Run structured medical report simplification with a trained adapter."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig

from config import DEFAULT_CONFIG_PATH, ProjectConfig, load_config
from evaluate import parse_generated_json
from utils import configure_logging, hardware_summary, set_reproducibility


LOGGER = logging.getLogger("medical_term_sft.inference")
SYSTEM_PROMPT = "You are an expert medical communication assistant."
USER_INSTRUCTION = "Simplify this medical report."


def parse_args() -> argparse.Namespace:
    """Parse inference arguments."""

    parser = argparse.ArgumentParser(
        description="Simplify a medical report with the trained Qwen3 adapter."
    )
    parser.add_argument("report", type=Path, help="UTF-8 medical report text file.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--adapter",
        type=Path,
        help="Adapter directory (default: outputs/best_adapter from config).",
    )
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _load_model(adapter_path: Path, config: ProjectConfig) -> Any:
    if not adapter_path.is_dir():
        raise FileNotFoundError(
            f"Adapter not found: {adapter_path}. Run train.py first or pass --adapter."
        )
    load_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available() and config.model.use_4bit:
        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    elif torch.backends.mps.is_available():
        load_kwargs["dtype"] = torch.float16
    else:
        load_kwargs["dtype"] = torch.float32
    return AutoPeftModelForCausalLM.from_pretrained(
        adapter_path, **load_kwargs
    ).eval()


def main() -> None:
    """Load one report, generate one answer, and print pretty JSON."""

    args = parse_args()
    configure_logging(args.verbose)
    config = load_config(args.config)
    set_reproducibility(config.training.seed, deterministic=False)
    hardware_summary()
    report = args.report.expanduser().resolve()
    if not report.is_file():
        raise FileNotFoundError(f"Report file not found: {report}")
    report_text = report.read_text(encoding="utf-8").strip()
    if not report_text:
        raise ValueError("The report file is empty.")

    adapter = (
        args.adapter.expanduser().resolve()
        if args.adapter
        else config.training.output_dir / "best_adapter"
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_model(adapter, config)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{USER_INSTRUCTION}\n\nMedical report:\n{report_text}",
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    )
    device = next(model.parameters()).device
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    max_new_tokens = args.max_new_tokens or config.generation.max_new_tokens
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(
        output[0, inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    )
    parsed = parse_generated_json(completion)
    if parsed.value is None:
        LOGGER.error("Model returned invalid JSON: %s", parsed.error)
        print(completion)
        sys.exit(2)
    print(json.dumps(parsed.value, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

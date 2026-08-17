import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation import Evaluator, aggregate_metrics
from performance import get_model_size_mb, timed_resource_block
from plots import create_plots
from utils import (
    LEVELS,
    ensure_output_dirs,
    load_config,
    medical_simplification_prompt,
    model_slug,
    setup_logging,
    write_json,
)


LOGGER = logging.getLogger(__name__)


def resolve_device(config: Dict[str, Any]) -> str:
    requested = config.get("evaluation", {}).get("device", "auto")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def resolve_dtype(config: Dict[str, Any]):
    dtype = config.get("generation", {}).get("torch_dtype", "auto")
    if dtype == "auto":
        return "auto"
    return getattr(torch, dtype)


def load_generation_model(model_name: str, config: Dict[str, Any]):
    generation_config = config.get("generation", {})
    device = resolve_device(config)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=generation_config.get("trust_remote_code", True),
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=resolve_dtype(config),
        trust_remote_code=generation_config.get("trust_remote_code", True),
    )
    model.to(device)
    model.eval()
    load_time = time.perf_counter() - start
    return tokenizer, model, load_time


def prepare_generation_inputs(tokenizer, model, prompt: str):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        except TypeError:
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
    else:
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True)

    if isinstance(encoded, torch.Tensor):
        encoded = {"input_ids": encoded}

    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    return input_ids, attention_mask


def generate_simplification(tokenizer, model, report: str, level: str, config: Dict[str, Any]) -> str:
    prompt = medical_simplification_prompt(report, level)
    generation_config = config.get("generation", {})

    input_ids, attention_mask = prepare_generation_inputs(tokenizer, model, prompt)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=generation_config.get("max_new_tokens", 768),
            temperature=generation_config.get("temperature", 0.2),
            top_p=generation_config.get("top_p", 0.9),
            do_sample=generation_config.get("do_sample", False),
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][input_ids.shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def benchmark_model(
    model_key: str,
    model_config: Dict[str, Any],
    input_df: pd.DataFrame,
    evaluator: Evaluator,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    model_name = model_config["model_name"]
    display_name = model_config.get("display_name", model_key)
    LOGGER.info("Loading model %s (%s)", display_name, model_name)
    tokenizer, model, load_time = load_generation_model(model_name, config)
    model_size = get_model_size_mb(model_name)
    rows: List[Dict[str, Any]] = []

    total = len(input_df) * len(LEVELS)
    progress = tqdm(total=total, desc=f"Benchmarking {model_key}")
    for _, source_row in input_df.iterrows():
        report_id = source_row["report_id"]
        report = str(source_row["report"])
        for level in LEVELS:
            row: Dict[str, Any] = {
                "report_id": report_id,
                "model_key": model_key,
                "model_name": model_name,
                "level": level,
                "original_report": report,
            }
            try:
                with timed_resource_block() as perf:
                    simplified = generate_simplification(tokenizer, model, report, level, config)
                row.update(perf)
                row["latency_seconds"] = perf["elapsed_seconds"]
                row["simplified_report"] = simplified
                row.update(evaluator.evaluate_row(report, simplified, level))
            except Exception as exc:
                LOGGER.exception("Benchmark failed for report_id=%s level=%s model=%s", report_id, level, model_key)
                row.update(
                    {
                        "simplified_report": "",
                        "error": repr(exc),
                        "latency_seconds": 0.0,
                        "peak_ram_mb": 0.0,
                        "peak_cpu_percent": 0.0,
                        "gpu_peak_vram_mb": 0.0,
                    }
                )
            row["model_loading_time_seconds"] = load_time
            row["model_size_mb"] = model_size
            rows.append(row)
            progress.update(1)
    progress.close()

    result_path = Path(config["paths"]["results_dir"]) / f"{model_slug(model_key)}_results.csv"
    pd.DataFrame(rows).to_csv(result_path, index=False)
    summary = aggregate_metrics(rows, config.get("readiness_weights", {}))
    summary.update(
        {
            "model_key": model_key,
            "model_name": model_name,
            "display_name": display_name,
            "results_csv": str(result_path),
        }
    )
    metrics_path = Path(config["paths"]["metrics_dir"]) / f"{model_slug(model_key)}_metrics.json"
    write_json(metrics_path, summary)
    LOGGER.info("Saved %s and %s", result_path, metrics_path)
    return summary


def build_report(metrics_by_model: Dict[str, Dict[str, Any]], report_path: str) -> None:
    rows = []
    for model_key, metrics in metrics_by_model.items():
        rows.append(
            {
                "Model": metrics.get("display_name", model_key),
                "Semantic Similarity": round(metrics.get("average_semantic_similarity", 0.0), 4),
                "Entity Recall": round(metrics.get("average_entity_recall", 0.0), 4),
                "FK Grade": round(metrics.get("average_flesch_kincaid_grade", 0.0), 2),
                "Hallucination Rate": round(metrics.get("hallucination_rate", 0.0), 4),
                "Critical Errors": metrics.get("critical_error_count", 0),
                "Avg Latency (s)": round(metrics.get("average_latency_seconds", 0.0), 3),
                "P95 Latency (s)": round(metrics.get("p95_latency_seconds", 0.0), 3),
                "Peak RAM (MB)": round(metrics.get("peak_ram_mb", 0.0), 1),
                "Peak VRAM (MB)": round(metrics.get("peak_gpu_vram_mb", 0.0), 1),
                "Readiness Score": metrics.get("overall_readiness_score", 0.0),
            }
        )
    table = pd.DataFrame(rows)
    best = table.sort_values("Readiness Score", ascending=False).head(1)
    markdown_table = dataframe_to_markdown(table)
    lines = [
        "# Medical Report Simplifier Benchmark",
        "",
        "## Model Comparison",
        "",
        markdown_table,
        "",
        "## Recommendation",
        "",
    ]
    if not best.empty:
        lines.append(
            f"Highest readiness score: **{best.iloc[0]['Model']}** "
            f"({best.iloc[0]['Readiness Score']}/100)."
        )
    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No benchmark rows were generated._"
    headers = [str(column) for column in df.columns]
    separator = ["---"] * len(headers)
    rows = []
    for _, row in df.iterrows():
        rows.append([str(row[column]) for column in df.columns])

    def line(values):
        escaped = [value.replace("|", "\\|").replace("\n", " ") for value in values]
        return "| " + " | ".join(escaped) + " |"

    return "\n".join([line(headers), line(separator), *(line(row) for row in rows)])


def validate_input_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"report_id", "report"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LLMs for medical report simplification.")
    parser.add_argument("--input", required=True, help="Input CSV with report_id,report columns.")
    parser.add_argument("--config", default="config.json", help="Benchmark config JSON path.")
    parser.add_argument("--models", nargs="*", default=None, help="Optional model keys to run.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    args = parser.parse_args()

    setup_logging(args.log_level)
    config = load_config(args.config)
    ensure_output_dirs(config)
    input_df = validate_input_csv(args.input)
    evaluator = Evaluator(config)

    selected_models = args.models or list(config["models"].keys())
    metrics_by_model: Dict[str, Dict[str, Any]] = {}
    for model_key in selected_models:
        if model_key not in config["models"]:
            raise KeyError(f"Model key {model_key!r} not found in config.")
        metrics_by_model[model_key] = benchmark_model(
            model_key, config["models"][model_key], input_df, evaluator, config
        )

    create_plots(metrics_by_model, config["paths"]["plots_dir"])
    build_report(metrics_by_model, config["paths"].get("report_path", "comparison_report.md"))
    LOGGER.info("Benchmark complete.")


if __name__ == "__main__":
    main()

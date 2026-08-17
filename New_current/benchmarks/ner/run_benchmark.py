"""Run identical annotated OCR text through all approved local NER candidates."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ner.contracts import ENTITY_TYPES, NormalizedEntity  # noqa: E402
from app.ner.errors import NERError  # noqa: E402
from app.ner.manifest import load_ner_model_manifest  # noqa: E402
from benchmarks.ner.providers import create_evaluation_registry  # noqa: E402
from benchmarks.ner.service import NERBenchmarkService  # noqa: E402

CANDIDATES = ("openmed-gliner", "biomedical-ner-all", "modernbert-biomedical-ner")


def create_ner_benchmark_service() -> NERBenchmarkService:
    """Compose evaluation candidates only when the offline runner is invoked."""
    manifest = load_ner_model_manifest()
    return NERBenchmarkService(create_evaluation_registry(manifest))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid benchmark configuration: {path}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark configuration must be an object: {path}.")
    return payload


def _configure_environment(configuration: dict[str, Any]) -> None:
    if configuration.get("schema_version") != "ner-evaluation-config-v1":
        raise ValueError("Unsupported NER evaluation configuration schema.")
    device = configuration.get("device")
    threshold = configuration.get("confidence_threshold")
    max_tokens = configuration.get("max_tokens")
    mappings = configuration.get("label_mappings")
    if device not in {"cpu", "cuda"}:
        raise ValueError("NER evaluation device must be cpu or cuda.")
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("NER confidence threshold must be between zero and one.")
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("NER max_tokens must be a positive integer.")
    if not isinstance(mappings, dict) or set(mappings) != set(CANDIDATES):
        raise ValueError("NER label mappings must define all candidates.")
    os.environ["NER_BENCHMARK_DEVICE"] = device
    os.environ["NER_BENCHMARK_CONFIDENCE_THRESHOLD"] = str(threshold)
    os.environ["NER_BENCHMARK_MAX_TOKENS"] = str(max_tokens)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    for candidate, mapping in mappings.items():
        variable = f"NER_{candidate.upper().replace('-', '_')}__LABEL_MAPPING_JSON"
        os.environ[variable] = json.dumps(mapping, sort_keys=True)


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on dataset line {line_number}.") from exc
        text = record.get("text")
        raw_entities = record.get("entities")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Dataset line {line_number} requires non-empty text.")
        if not isinstance(raw_entities, list):
            raise ValueError(f"Dataset line {line_number} requires an entities list.")
        references = tuple(_reference_entity(text, item) for item in raw_entities)
        records.append({**record, "references": references})
    if not records:
        raise ValueError("Benchmark dataset is empty.")
    represented = {entity.label for record in records for entity in record["references"]}
    if represented != set(ENTITY_TYPES):
        missing = sorted(set(ENTITY_TYPES) - represented)
        raise ValueError(f"Benchmark dataset does not cover canonical labels: {missing}.")
    return records


def _reference_entity(text: str, item: Any) -> NormalizedEntity:
    if not isinstance(item, dict):
        raise ValueError("Benchmark entities must be objects.")
    entity = NormalizedEntity(
        str(item["text"]),
        str(item["label"]),
        int(item["start"]),
        int(item["end"]),
        float(item.get("confidence", 1.0)),
    )
    if text[entity.start : entity.end] != entity.text:
        raise ValueError(f"Reference entity offsets do not match text: {entity.text!r}.")
    return entity


async def run(dataset: Path, output: Path, configuration_path: Path) -> dict[str, Any]:
    configuration = _load_json(configuration_path)
    _configure_environment(configuration)
    records = _load_dataset(dataset)
    manifest = load_ner_model_manifest()
    results: dict[str, Any] = {}
    for candidate in CANDIDATES:
        service = create_ner_benchmark_service()
        documents: list[dict[str, Any]] = []
        try:
            health = {item.provider_name: item for item in service.models()}[candidate]
            if health.status.value == "not_configured":
                results[candidate] = {
                    "status": "NOT VERIFIED",
                    "reason": health.detail,
                    "documents": [],
                }
                continue
            for index, record in enumerate(records):
                result = await service.benchmark(candidate, record["text"], record["references"])
                documents.append(
                    {
                        "document_id": record.get("document_id", str(index)),
                        "document_index": index,
                        "reference_entities": [asdict(item) for item in record["references"]],
                        "entities": [asdict(item) for item in result.entities],
                        "metrics": asdict(result.metrics),
                        "warnings": list(result.warnings),
                    }
                )
            results[candidate] = {
                "status": "PASS",
                "model": _model_inventory(manifest.candidates[candidate], health.metadata),
                "documents": documents,
                "per_entity_metrics": _per_entity_metrics(documents),
                "summary": _summary(documents),
            }
        except NERError as exc:
            results[candidate] = {
                "status": "FAIL",
                "reason": exc.message,
                "documents": documents,
            }
        finally:
            await service.shutdown()
    leaderboard = _leaderboard(results)
    recommended = leaderboard[0]["candidate"] if len(leaderboard) == len(CANDIDATES) else None
    payload = {
        "schema_version": "ner-benchmark-report-v2",
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": str(dataset.resolve()),
        "dataset_records": len(records),
        "configuration": configuration,
        "runtime": {
            "platform": platform.platform(),
            "python": sys.version.splitlines()[0],
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "winner": None,
        "recommended_candidate": recommended,
        "recommendation_basis": (
            "Highest macro F1 across all eight canonical entity types; overall F1 and "
            "mean inference latency are deterministic tie-breakers."
            if recommended
            else "No recommendation because every candidate did not complete successfully."
        ),
        "leaderboard": leaderboard,
        "candidates": results,
    }
    _write_reports(payload, output)
    return payload


def _model_inventory(entry: Any, metadata: Any) -> dict[str, Any]:
    cache_path = Path(metadata.configuration["cache_path"])
    observed_sha256, file_count, total_bytes = _directory_sha256(cache_path)
    return {
        "repository_id": entry.repository_id,
        "revision": entry.pinned_revision,
        "license": entry.license,
        "local_cache": metadata.configuration["cache_path"],
        "observed_tree_sha256": observed_sha256,
        "artifact_file_count": file_count,
        "artifact_bytes": total_bytes,
        "framework": metadata.framework,
        "device": metadata.device,
        "configuration": metadata.configuration,
    }


def _directory_sha256(cache_path: Path) -> tuple[str, int, int]:
    """Hash model artifacts deterministically without including HF bookkeeping."""
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    files = sorted(
        path
        for path in cache_path.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(cache_path).parts
    )
    for path in files:
        relative = path.relative_to(cache_path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        file_count += 1
        total_bytes += size
    return digest.hexdigest(), file_count, total_bytes


def _entity_sets(
    documents: list[dict[str, Any]], label: str | None = None
) -> tuple[set[tuple[Any, ...]], set[tuple[Any, ...]]]:
    predicted: set[tuple[Any, ...]] = set()
    references: set[tuple[Any, ...]] = set()
    for document in documents:
        index = document["document_index"]
        for entity in document["entities"]:
            if label is None or entity["label"] == label:
                predicted.add((index, entity["start"], entity["end"], entity["label"]))
        for entity in document["reference_entities"]:
            if label is None or entity["label"] == label:
                references.add((index, entity["start"], entity["end"], entity["label"]))
    return predicted, references


def _quality_metrics(
    predicted: set[tuple[Any, ...]], references: set[tuple[Any, ...]]
) -> dict[str, Any]:
    true_positive = len(predicted & references)
    false_positive = len(predicted - references)
    false_negative = len(references - predicted)
    precision = true_positive / (true_positive + false_positive) if predicted else 0.0
    recall = true_positive / len(references) if references else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = true_positive + false_positive + false_negative
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1_score, 6),
        "entity_level_accuracy": round(true_positive / total, 6) if total else 1.0,
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "reference_count": len(references),
        "predicted_count": len(predicted),
    }


def _per_entity_metrics(documents: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {label: _quality_metrics(*_entity_sets(documents, label)) for label in ENTITY_TYPES}


def _summary(documents: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _quality_metrics(*_entity_sets(documents))
    per_entity = _per_entity_metrics(documents)
    summary["macro_f1_score"] = round(
        mean(metrics["f1_score"] for metrics in per_entity.values()), 6
    )
    for metric in ("inference_latency_ms", "tokens_per_second"):
        values = [
            document["metrics"][metric]
            for document in documents
            if document["metrics"][metric] is not None
        ]
        summary[f"mean_{metric}"] = round(mean(values), 6) if values else None
    for metric in ("peak_ram_mb", "peak_gpu_memory_mb"):
        values = [
            document["metrics"][metric]
            for document in documents
            if document["metrics"][metric] is not None
        ]
        summary[metric] = round(max(values), 6) if values else None
    loading = [
        document["metrics"]["model_loading_time_ms"]
        for document in documents
        if document["metrics"]["model_loading_time_ms"] is not None
    ]
    summary["model_loading_time_ms"] = loading[0] if loading else None
    return summary


def _leaderboard(results: dict[str, Any]) -> list[dict[str, Any]]:
    completed = [
        {
            "candidate": candidate,
            "macro_f1_score": result["summary"]["macro_f1_score"],
            "overall_f1_score": result["summary"]["f1_score"],
            "mean_inference_latency_ms": result["summary"]["mean_inference_latency_ms"],
        }
        for candidate, result in results.items()
        if result["status"] == "PASS"
    ]
    completed.sort(
        key=lambda item: (
            -item["macro_f1_score"],
            -item["overall_f1_score"],
            item["mean_inference_latency_ms"],
        )
    )
    return [{"rank": rank, **item} for rank, item in enumerate(completed, 1)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...] = ()) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        names = tuple(rows[0]) if rows else fieldnames
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _write_reports(payload: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "ner_benchmark_report.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    overall_rows = []
    entity_rows = []
    for candidate, result in payload["candidates"].items():
        summary = result.get("summary", {})
        overall_rows.append(
            {
                "candidate": candidate,
                "status": result["status"],
                **summary,
                "reason": result.get("reason"),
            }
        )
        for label, metrics in result.get("per_entity_metrics", {}).items():
            entity_rows.append({"candidate": candidate, "entity_type": label, **metrics})
    _write_csv(output / "ner_benchmark_metrics.csv", overall_rows)
    _write_csv(
        output / "ner_benchmark_per_entity.csv",
        entity_rows,
        ("candidate", "entity_type"),
    )
    _write_csv(
        output / "ner_benchmark_leaderboard.csv",
        payload["leaderboard"],
        (
            "rank",
            "candidate",
            "macro_f1_score",
            "overall_f1_score",
            "mean_inference_latency_ms",
        ),
    )
    (output / "ner_benchmark_report.md").write_text(_markdown_report(payload), encoding="utf-8")


def _markdown_report(payload: dict[str, Any]) -> str:
    overall_header = (
        "| Candidate | Status | Precision | Recall | F1 | Accuracy | FP | FN | "
        "Load ms | Latency ms | RAM MiB | GPU MiB | Tokens/s |"
    )
    leaderboard = "\n".join(
        f"| {row['rank']} | {row['candidate']} | {row['macro_f1_score']:.6f} | "
        f"{row['overall_f1_score']:.6f} | {row['mean_inference_latency_ms']:.3f} |"
        for row in payload["leaderboard"]
    )
    overall = "\n".join(
        _overall_markdown_row(candidate, result)
        for candidate, result in payload["candidates"].items()
    )
    per_entity_sections = []
    for candidate, result in payload["candidates"].items():
        rows = "\n".join(
            f"| {label} | {metrics['precision']:.6f} | {metrics['recall']:.6f} | "
            f"{metrics['f1_score']:.6f} | {metrics['entity_level_accuracy']:.6f} | "
            f"{metrics['false_positives']} | {metrics['false_negatives']} |"
            for label, metrics in result.get("per_entity_metrics", {}).items()
        )
        per_entity_sections.append(
            f"### {candidate}\n\n"
            "| Entity type | Precision | Recall | F1 | Accuracy | FP | FN |\n"
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
            f"{rows or 'No successful benchmark output.'}"
        )
    recommendation = payload["recommended_candidate"] or "NOT VERIFIED"
    return f"""# Medical NER Candidate Benchmark

Generated: {payload["generated_at"]}

Dataset: `{payload["dataset"]}` ({payload["dataset_records"]} synthetic de-identified records)  
CUDA available: `{payload["runtime"]["cuda_available"]}`  
Approved winner: **None**

## Recommendation

**{recommendation}** is recommended for explicit production-model review. It has not been
approved or integrated. Basis: {payload["recommendation_basis"]}

## Leaderboard

| Rank | Candidate | Macro F1 | Overall F1 | Mean latency ms |
| ---: | --- | ---: | ---: | ---: |
{leaderboard or "| - | NOT VERIFIED | - | - | - |"}

## Overall metrics

{overall_header}
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{overall}

## Per-entity metrics

{chr(10).join(per_entity_sections)}

## Evidence limits

- Every candidate used the same synthetic, de-identified normalized OCR text and
  exact-span references.
- GPU memory is NOT VERIFIED when CUDA is unavailable; it is never reported as zero.
- This benchmark is evidence for a recommendation, not production approval or integration.
"""


def _overall_markdown_row(candidate: str, result: dict[str, Any]) -> str:
    summary = result.get("summary")
    if summary is None:
        return f"| {candidate} | {result['status']} | - | - | - | - | - | - | - | - | - | - | - |"
    gpu = summary["peak_gpu_memory_mb"]
    gpu_text = f"{gpu:.3f}" if gpu is not None else "NOT VERIFIED"
    return (
        f"| {candidate} | {result['status']} | {summary['precision']:.6f} | "
        f"{summary['recall']:.6f} | {summary['f1_score']:.6f} | "
        f"{summary['entity_level_accuracy']:.6f} | {summary['false_positives']} | "
        f"{summary['false_negatives']} | {summary['model_loading_time_ms']:.3f} | "
        f"{summary['mean_inference_latency_ms']:.3f} | {summary['peak_ram_mb']:.3f} | "
        f"{gpu_text} | {summary['mean_tokens_per_second']:.3f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=Path(__file__).with_name("dataset_template.jsonl")
    )
    parser.add_argument(
        "--configuration",
        type=Path,
        default=Path(__file__).with_name("evaluation_config.json"),
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("reports"))
    arguments = parser.parse_args()
    payload = asyncio.run(run(arguments.dataset, arguments.output, arguments.configuration))
    print(
        json.dumps(
            {
                "output": str(arguments.output.resolve()),
                "recommended_candidate": payload["recommended_candidate"],
                "winner": payload["winner"],
            }
        )
    )


if __name__ == "__main__":
    main()

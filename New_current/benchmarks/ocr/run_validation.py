"""Run evidence-based validation for the Phase 2 OCR providers.

The runner never downloads or selects a model. It uses only immutable provider
configuration supplied through the production environment. Missing prerequisites are
reported as NOT VERIFIED and metric values remain null.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any

import fitz
import psutil
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ocr.postprocessing.stages import (  # noqa: E402
    MedicalAbbreviationStage,
    RegexNormalizationStage,
    SymSpellStage,
)
from app.ocr.providers.contracts import ProviderDocument, ProviderKind  # noqa: E402
from app.ocr.providers.documents import decode_document_pages  # noqa: E402
from app.ocr.providers.errors import (  # noqa: E402
    ProviderConfigurationError,
    ProviderError,
    ProviderInitializationError,
    ProviderUnavailableError,
)
from app.ocr.providers.implementations import register_builtin_providers  # noqa: E402
from app.ocr.providers.manifest import (  # noqa: E402
    PENDING_APPROVAL,
    load_model_manifest,
    merge_manifest_model_configuration,
)
from app.ocr.providers.registry import ProviderRegistry  # noqa: E402
from app.ocr.providers.runtime import (  # noqa: E402
    OCRProviderFactory,
    PostProcessorFactory,
)

REPORTS_DIRECTORY = PROJECT_ROOT / "benchmarks" / "ocr" / "reports"
FIXTURE_DIRECTORY = PROJECT_ROOT / "tests" / "ocr" / "fixtures"
STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_NOT_VERIFIED = "NOT VERIFIED"

SELECTION_VARIABLES = {
    ProviderKind.OCR: "OCR_PROVIDER",
    ProviderKind.POSTPROCESSOR: "POSTPROCESSOR_PROVIDER",
}
CONFIG_PREFIXES = {
    ProviderKind.OCR: "OCR_CONFIG__",
    ProviderKind.POSTPROCESSOR: "POSTPROCESSOR_CONFIG__",
}


class EvidenceLogHandler(logging.Handler):
    """Capture privacy-safe structured fields emitted by providers."""

    _standard = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}

    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in self._standard and _is_json_value(value)
        }
        self.records.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                **fields,
            }
        )


def _is_json_value(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


@contextmanager
def measured_memory() -> Any:
    process = psutil.Process()
    before = process.memory_info().rss
    measurement = {"rss_before_mb": _mb(before), "rss_after_mb": None, "rss_delta_mb": None}
    yield measurement
    after = process.memory_info().rss
    measurement["rss_after_mb"] = _mb(after)
    measurement["rss_delta_mb"] = round(_mb(after - before), 3)


def _mb(value: int) -> float:
    return round(value / (1024 * 1024), 3)


def _provider_configuration(kind: ProviderKind) -> tuple[str | None, dict[str, str]]:
    selection = os.getenv(SELECTION_VARIABLES[kind], "").strip().casefold() or None
    prefix = CONFIG_PREFIXES[kind]
    configuration = {
        key.removeprefix(prefix).casefold(): value.strip()
        for key, value in os.environ.items()
        if key.startswith(prefix)
    }
    if kind is ProviderKind.OCR:
        with suppress(ProviderConfigurationError):
            configuration = merge_manifest_model_configuration(
                kind,
                configuration,
                manifest=load_model_manifest(),
            )
    return selection, configuration


def _benchmark_postprocessor_configuration() -> dict[str, str]:
    return {
        "provider_version": "benchmark-fixture-v1",
        "dictionary_path": str(FIXTURE_DIRECTORY / "symspell_frequency.txt"),
        "dictionary_version": "synthetic-benchmark-v1",
        "abbreviation_dictionary_path": str(FIXTURE_DIRECTORY / "medical_abbreviations.json"),
        "abbreviation_dictionary_version": "synthetic-benchmark-v1",
        "protected_terms_path": str(FIXTURE_DIRECTORY / "protected_medical_terms.json"),
        "rule_version": "synthetic-benchmark-v1",
        "max_edit_distance": "2",
        "prefix_length": "7",
        "min_word_length": "4",
        "min_frequency": "1",
    }


def _synthetic_documents() -> dict[str, ProviderDocument]:
    text = "Synthetic de-identified report BP 120 / 80 haemoglobn 6 . 5 %"

    def image_bytes(image_format: str) -> bytes:
        buffer = BytesIO()
        image = Image.new("RGB", (900, 220), "white")
        ImageDraw.Draw(image).text((30, 90), text, fill="black")
        image.save(buffer, format=image_format)
        return buffer.getvalue()

    pdf = fitz.open()
    for page_number in range(1, 3):
        page = pdf.new_page(width=900, height=220)
        page.insert_text((30, 110), f"{text} page {page_number}", fontsize=18)
    try:
        multipage_pdf = pdf.tobytes()
    finally:
        pdf.close()

    single_pdf = fitz.open()
    page = single_pdf.new_page(width=900, height=220)
    page.insert_text((30, 110), text, fontsize=18)
    try:
        single_page_pdf = single_pdf.tobytes()
    finally:
        single_pdf.close()

    tiff_buffer = BytesIO()
    first = Image.new("RGB", (900, 220), "white")
    second = Image.new("RGB", (900, 220), "white")
    ImageDraw.Draw(first).text((30, 90), f"{text} page 1", fill="black")
    ImageDraw.Draw(second).text((30, 90), f"{text} page 2", fill="black")
    first.save(tiff_buffer, format="TIFF", save_all=True, append_images=[second])

    return {
        "pdf": ProviderDocument(single_page_pdf, "pdf", "synthetic.pdf"),
        "multi_page_pdf": ProviderDocument(multipage_pdf, "pdf", "synthetic-multi.pdf"),
        "png": ProviderDocument(image_bytes("PNG"), "png", "synthetic.png"),
        "jpeg": ProviderDocument(image_bytes("JPEG"), "jpeg", "synthetic.jpeg"),
        "tiff": ProviderDocument(tiff_buffer.getvalue(), "tiff", "synthetic.tiff"),
    }


def _requirement_rows() -> list[dict[str, str]]:
    return [
        {"requirement": requirement, "status": STATUS_NOT_VERIFIED, "evidence": evidence}
        for requirement, evidence in (
            ("Model manifest verification", "Manifest has not been inspected."),
            ("Qwen3-VL loads correctly", "No live initialization attempt completed."),
            ("OCR runs on CPU", "No live CPU inference completed."),
            ("OCR runs on CUDA (if available)", "No live CUDA inference completed."),
            ("PDF", "Format decoding and live Qwen3-VL inference have not both passed."),
            (
                "Multi-page PDF",
                "Ordered decoding and live Qwen3-VL inference have not both passed.",
            ),
            ("PNG", "Format decoding and live Qwen3-VL inference have not both passed."),
            ("JPEG", "Format decoding and live Qwen3-VL inference have not both passed."),
            ("TIFF", "Format decoding and live Qwen3-VL inference have not both passed."),
            ("Confidence generation", "No live model confidence distribution was measured."),
            ("Processing time", "No live model inference timing was measured."),
            ("OCR postprocessing", "Post-processing benchmark has not run."),
            ("Structured logging", "Structured provider records have not been inspected."),
        )
    ]


def _set_requirement(
    rows: list[dict[str, str]], requirement: str, status: str, evidence: str
) -> None:
    row = next(item for item in rows if item["requirement"] == requirement)
    row.update(status=status, evidence=evidence)


def _validate_decoding(documents: dict[str, ProviderDocument]) -> dict[str, Any]:
    results = {}
    for name, document in documents.items():
        try:
            pages = decode_document_pages(
                document,
                pdf_render_dpi=96,
                max_image_size=1024,
                max_pages=10,
            )
            results[name] = {
                "status": STATUS_PASS,
                "page_count": len(pages),
                "page_order": [page.page_number for page in pages],
            }
        except ProviderError as exc:
            results[name] = {"status": STATUS_FAIL, "error": exc.message}
    return results


async def _validate_postprocessor(
    registry: ProviderRegistry, error_examples: list[dict[str, Any]]
) -> dict[str, Any]:
    selected, configured = _provider_configuration(ProviderKind.POSTPROCESSOR)
    source = "deployment_environment"
    if selected is None:
        selected = "symspell"
        configured = _benchmark_postprocessor_configuration()
        source = "synthetic_benchmark_fixture"
    provider = PostProcessorFactory(registry).create(selected, configured)
    sample = "BP 120 / 80  haemoglobn 6 . 5 % Metformin"
    started = perf_counter()
    await provider.initialize()
    try:
        result = provider.normalize(sample, document_type="printed_image")
        stages = [
            {
                "stage": stage.stage,
                "latency_ms": stage.latency_ms,
                "corrections": stage.corrections,
                "configuration": dict(stage.configuration),
            }
            for stage in result.stages
        ]
        error_examples.extend(_postprocessing_examples(configured, sample))
        return {
            "status": STATUS_PASS,
            "configuration_source": source,
            "initialization_time_ms": round((perf_counter() - started) * 1000, 3),
            "processing_time_ms": result.processing_time_ms,
            "total_corrections": result.total_corrections,
            "stages": stages,
            "normalized_output": result.normalized_text,
        }
    finally:
        await provider.shutdown()


def _postprocessing_examples(
    configuration: dict[str, str], source_text: str
) -> list[dict[str, Any]]:
    regex = RegexNormalizationStage(rule_version=configuration["rule_version"])
    abbreviations = MedicalAbbreviationStage(
        dictionary_path=configuration["abbreviation_dictionary_path"],
        dictionary_version=configuration["abbreviation_dictionary_version"],
    )
    symspell = SymSpellStage(
        dictionary_path=configuration["dictionary_path"],
        dictionary_version=configuration["dictionary_version"],
        max_edit_distance=int(configuration["max_edit_distance"]),
        prefix_length=int(configuration["prefix_length"]),
        min_word_length=int(configuration["min_word_length"]),
        min_frequency=int(configuration["min_frequency"]),
        protected_terms=abbreviations.protected_terms,
        protected_terms_path=configuration["protected_terms_path"],
    )
    examples = []
    current = source_text
    for name, stage in (
        ("regex", regex),
        ("medical_abbreviation_dictionary", abbreviations),
        ("symspell", symspell),
    ):
        result = stage.process(current)
        examples.append(
            {
                "category": name,
                "before": current,
                "after": result.text,
                "corrections": result.corrections,
                "latency_ms": result.latency_ms,
            }
        )
        current = result.text
    return examples


async def _benchmark_ocr(
    registry: ProviderRegistry,
    documents: dict[str, ProviderDocument],
    device: str,
    warm_runs: int,
    error_examples: list[dict[str, Any]],
) -> dict[str, Any]:
    selected, configuration = _provider_configuration(ProviderKind.OCR)
    if selected is None or not configuration:
        return {"status": STATUS_NOT_VERIFIED, "reason": "Provider environment is absent."}
    configuration["device"] = device
    provider = OCRProviderFactory(registry).create(selected, configuration)
    try:
        with measured_memory() as load_memory:
            started = perf_counter()
            await provider.initialize()
            load_ms = round((perf_counter() - started) * 1000, 3)
        outputs: dict[str, Any] = {}
        first_started = perf_counter()
        first = provider.process(documents["png"])
        first_latency_ms = round((perf_counter() - first_started) * 1000, 3)
        warm_latencies = []
        for _ in range(warm_runs):
            started = perf_counter()
            provider.process(documents["png"])
            warm_latencies.append(round((perf_counter() - started) * 1000, 3))
        for name, document in documents.items():
            try:
                with measured_memory() as inference_memory:
                    result = provider.process(document)
            except (
                ProviderConfigurationError,
                ProviderInitializationError,
                ProviderUnavailableError,
            ) as exc:
                outputs[name] = {"status": STATUS_NOT_VERIFIED, "reason": exc.message}
                continue
            except ProviderError as exc:
                outputs[name] = {"status": STATUS_FAIL, "error": exc.message}
                continue
            outputs[name] = {
                "status": STATUS_PASS,
                "page_count": len(result.pages),
                "page_order": [page.page_number for page in result.pages],
                "processing_time_ms": result.processing_time_ms,
                "pages_per_second": result.statistics.pages_per_second,
                "confidence": result.confidence,
                "confidence_method": result.confidence_method,
                "gpu_memory_allocated_mb": result.statistics.gpu_memory_allocated_mb,
                "gpu_peak_memory_mb": result.statistics.gpu_peak_memory_mb,
                "cpu_memory": inference_memory,
            }
            error_examples.append(
                {
                    "category": "ocr_output_without_ground_truth",
                    "document": name,
                    "status": STATUS_NOT_VERIFIED,
                    "reason": (
                        "No approved ground-truth corpus was supplied; OCR mistakes "
                        "cannot be scored."
                    ),
                }
            )
        overall_status = (
            STATUS_PASS
            if all(item.get("status") == STATUS_PASS for item in outputs.values())
            else STATUS_FAIL
        )
        return {
            "status": overall_status,
            "device": device,
            "model_loading_time_ms": load_ms,
            "first_inference_latency_ms": first_latency_ms,
            "warm_inference_latency_ms": warm_latencies,
            "warm_inference_mean_ms": round(sum(warm_latencies) / len(warm_latencies), 3),
            "load_memory": load_memory,
            "first_confidence": first.confidence,
            "metadata": _safe_metadata(first.provider_metadata),
            "formats": outputs,
        }
    except (
        ProviderConfigurationError,
        ProviderInitializationError,
        ProviderUnavailableError,
    ) as exc:
        return {"status": STATUS_NOT_VERIFIED, "device": device, "reason": exc.message}
    except ProviderError as exc:
        return {"status": STATUS_FAIL, "device": device, "error": exc.message}
    finally:
        await provider.shutdown()


def _safe_metadata(metadata: Any) -> dict[str, Any] | None:
    if metadata is None:
        return None
    return {
        "provider_name": metadata.provider_name,
        "provider_version": metadata.provider_version,
        "provider_kind": metadata.provider_kind.value,
        "supported_file_types": list(metadata.supported_file_types),
        "supported_document_types": list(metadata.supported_document_types),
        "configuration": dict(metadata.configuration),
        "startup_timestamp": (
            metadata.startup_timestamp.isoformat() if metadata.startup_timestamp else None
        ),
    }


def _runtime_inventory() -> dict[str, Any]:
    cache_root = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
    cached_model_candidates = sorted(
        path.name
        for path in cache_root.glob("models--*")
        if "qwen3" in path.name.casefold() and "vl" in path.name.casefold()
    )
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "cuda_available_to_torch": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "physical_gpu": _physical_gpu_inventory(),
        "cached_qwen3_vl_candidates": cached_model_candidates,
        "provider_environment_present": {
            kind.value: bool(_provider_configuration(kind)[0])
            and bool(_provider_configuration(kind)[1])
            for kind in ProviderKind
        },
    }


def _validate_model_manifest() -> dict[str, Any]:
    try:
        manifest = load_model_manifest()
    except ProviderConfigurationError as exc:
        return {
            "status": STATUS_NOT_VERIFIED,
            "evidence": exc.message,
            "path": None,
            "models": {},
        }
    results: dict[str, Any] = {}
    for kind in (ProviderKind.OCR,):
        entry = manifest.entry_for(kind)
        selected, configured = _provider_configuration(kind)
        configured_repository = configured.get("model_name")
        configured_revision = configured.get("model_revision")
        cache_path = (manifest.path.parent / entry.local_cache_path).resolve()
        direct_revision_record = (
            cache_path / ".cache" / "huggingface" / "trees" / f"{entry.pinned_revision}.json"
        )
        revision_candidates = (
            [cache_path]
            if (cache_path / "config.json").is_file() and direct_revision_record.is_file()
            else [cache_path]
            if cache_path.is_dir() and cache_path.name == entry.pinned_revision
            else (
                [path for path in cache_path.rglob(entry.pinned_revision) if path.is_dir()]
                if cache_path.is_dir() and entry.pinned_revision != PENDING_APPROVAL
                else []
            )
        )
        revision_paths = [
            path for path in revision_candidates if any(item.is_file() for item in path.rglob("*"))
        ]
        identity_approved = all(
            value != PENDING_APPROVAL
            for value in (entry.repository_id, entry.pinned_revision, entry.license)
        )
        configuration_matches = bool(selected) and selected == entry.provider
        if identity_approved:
            configuration_matches = configuration_matches and (
                configured_repository == entry.repository_id
                and configured_revision == entry.pinned_revision
            )
        checksum_status = "NOT PROVIDED"
        observed_sha256 = None
        if entry.expected_sha256 != PENDING_APPROVAL and revision_paths:
            observed_sha256 = _directory_sha256(revision_paths[0])
            checksum_status = (
                STATUS_PASS if observed_sha256 == entry.expected_sha256 else STATUS_FAIL
            )
        if not identity_approved or not revision_paths or not configuration_matches:
            status = STATUS_NOT_VERIFIED
        elif checksum_status == STATUS_FAIL:
            status = STATUS_FAIL
        else:
            status = STATUS_PASS
        results[entry.key] = {
            "status": status,
            "provider": entry.provider,
            "repository_id": entry.repository_id,
            "pinned_revision": entry.pinned_revision,
            "license": entry.license,
            "expected_sha256": entry.expected_sha256,
            "observed_sha256": observed_sha256,
            "checksum_status": checksum_status,
            "cache_path": str(cache_path),
            "revision_available": bool(revision_paths),
            "provider_configuration_matches": configuration_matches,
        }
    statuses = {result["status"] for result in results.values()}
    overall = (
        STATUS_FAIL
        if STATUS_FAIL in statuses
        else STATUS_PASS
        if statuses == {STATUS_PASS}
        else STATUS_NOT_VERIFIED
    )
    pending = [key for key, result in results.items() if result["status"] != STATUS_PASS]
    evidence = (
        "Every manifest model, immutable revision, provider configuration, and local "
        "snapshot was verified."
        if overall == STATUS_PASS
        else "Awaiting approved model identifiers and checkpoints: " + ", ".join(pending)
    )
    return {
        "status": overall,
        "evidence": evidence,
        "path": str(manifest.path),
        "schema_version": manifest.schema_version,
        "models": results,
    }


def _directory_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _physical_gpu_inventory() -> list[str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _apply_requirement_evidence(payload: dict[str, Any]) -> None:
    rows = payload["requirements"]
    manifest = payload["model_manifest"]
    _set_requirement(
        rows,
        "Model manifest verification",
        manifest["status"],
        manifest["evidence"],
    )
    ocr = payload["benchmarks"]["ocr"]
    cpu_ocr = ocr.get("cpu", {})
    cuda_ocr = ocr.get("cuda", {})
    _set_requirement(
        rows,
        "Qwen3-VL loads correctly",
        cpu_ocr.get("status", STATUS_NOT_VERIFIED),
        _result_evidence(cpu_ocr),
    )
    _set_requirement(
        rows,
        "OCR runs on CPU",
        cpu_ocr.get("status", STATUS_NOT_VERIFIED),
        _result_evidence(cpu_ocr),
    )
    cuda_status = cuda_ocr.get("status", STATUS_NOT_VERIFIED)
    if not payload["runtime"]["cuda_available_to_torch"]:
        cuda_status = STATUS_NOT_VERIFIED
    _set_requirement(
        rows,
        "OCR runs on CUDA (if available)",
        cuda_status,
        _result_evidence(cuda_ocr),
    )
    for requirement, key in (
        ("PDF", "pdf"),
        ("Multi-page PDF", "multi_page_pdf"),
        ("PNG", "png"),
        ("JPEG", "jpeg"),
        ("TIFF", "tiff"),
    ):
        live_format = cpu_ocr.get("formats", {}).get(key, {})
        decoding = payload["format_decoding"].get(key, {})
        status = live_format.get("status", STATUS_NOT_VERIFIED)
        live_evidence = _result_evidence(live_format or cpu_ocr)
        evidence = (
            f"Decoder {decoding.get('status', STATUS_NOT_VERIFIED)}; live Qwen3-VL {live_evidence}"
        )
        _set_requirement(rows, requirement, status, evidence)
    confidence_values = [
        result.get("confidence")
        for result in cpu_ocr.get("formats", {}).values()
        if result.get("confidence") is not None
    ]
    _set_requirement(
        rows,
        "Confidence generation",
        STATUS_PASS if confidence_values else STATUS_NOT_VERIFIED,
        (
            f"Measured {len(confidence_values)} live confidence values."
            if confidence_values
            else "No live model confidence values were produced."
        ),
    )
    timing_values = [
        result.get("processing_time_ms")
        for result in cpu_ocr.get("formats", {}).values()
        if result.get("processing_time_ms") is not None
    ]
    _set_requirement(
        rows,
        "Processing time",
        STATUS_PASS if timing_values else STATUS_NOT_VERIFIED,
        (
            f"Measured {len(timing_values)} live inference timings."
            if timing_values
            else "No live model inference timings were produced."
        ),
    )
    postprocessor = payload["benchmarks"]["postprocessor"]
    _set_requirement(
        rows,
        "OCR postprocessing",
        postprocessor.get("status", STATUS_FAIL),
        (
            f"Synthetic pipeline completed with {postprocessor.get('total_corrections', 0)} "
            f"corrections in {postprocessor.get('processing_time_ms')} ms."
            if postprocessor.get("status") == STATUS_PASS
            else _result_evidence(postprocessor)
        ),
    )
    structured_records = [
        record
        for record in payload["structured_logs"]
        if record.get("event") in {"provider_initialized", "postprocessing_completed"}
        and record.get("provider_name")
        and record.get("provider_version")
    ]
    _set_requirement(
        rows,
        "Structured logging",
        STATUS_PASS if structured_records else STATUS_FAIL,
        (
            f"Captured {len(structured_records)} provider records with structured "
            "lifecycle/stage fields."
        ),
    )


def _result_evidence(result: dict[str, Any]) -> str:
    return str(result.get("reason") or result.get("error") or result.get("status", "did not run"))


def _confidence_distribution(payload: dict[str, Any]) -> dict[str, Any]:
    values = []
    for run in payload["benchmarks"]["ocr"].values():
        for result in run.get("formats", {}).values():
            if result.get("confidence") is not None:
                values.append(float(result["confidence"]))
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "values": []}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "values": values,
    }


def _screenshot_inventory(output_directory: Path) -> dict[str, dict[str, str]]:
    screenshot_directory = output_directory / "screenshots"
    evidence = {
        "dashboard": (
            "developer-dashboard.png",
            STATUS_PASS,
            "The existing developer page rendered on the screenshot-only server.",
        ),
        "swagger": (
            "swagger.png",
            STATUS_PASS,
            "Swagger rendered and exposed GET /api/v1/ocr/health.",
        ),
        "health": (
            "health-endpoint-503.png",
            STATUS_NOT_VERIFIED,
            "The actual health operation returned 503 because providers were not initialized.",
        ),
        "ocr_output": (
            "ocr-output-not-verified.png",
            STATUS_NOT_VERIFIED,
            "A synthetic upload produced no OCR output because the runtime was not ready.",
        ),
    }
    result = {}
    for name, (filename, status, detail) in evidence.items():
        path = screenshot_directory / filename
        result[name] = {
            "status": status if path.is_file() else STATUS_NOT_VERIFIED,
            "path": str(path),
            "evidence": detail if path.is_file() else "Not captured.",
        }
    return result


def _write_reports(payload: dict[str, Any], output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "validation_report.json"
    csv_path = output_directory / "benchmark_metrics.csv"
    requirements_path = output_directory / "validation_requirements.csv"
    markdown_path = output_directory / "validation_report.md"
    error_path = output_directory / "error_report.md"
    logs_path = output_directory / "structured_logs.jsonl"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with requirements_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("requirement", "status", "evidence"))
        writer.writeheader()
        writer.writerows(payload["requirements"])
    metric_rows = _metric_rows(payload)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("provider", "device", "document", "metric", "value", "unit", "status"),
        )
        writer.writeheader()
        writer.writerows(metric_rows)
    markdown_path.write_text(_validation_markdown(payload), encoding="utf-8")
    error_path.write_text(_error_markdown(payload), encoding="utf-8")
    logs_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in payload["structured_logs"]
        ),
        encoding="utf-8",
    )


def _write_synthetic_fixtures(
    documents: dict[str, ProviderDocument], output_directory: Path
) -> None:
    fixture_output = output_directory / "synthetic_inputs"
    fixture_output.mkdir(parents=True, exist_ok=True)
    for document in documents.values():
        if document.filename:
            (fixture_output / document.filename).write_bytes(document.content)


def _metric_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        provider: str,
        device: str,
        document: str,
        metric: str,
        value: Any,
        unit: str,
        status: str,
    ) -> None:
        rows.append(
            {
                "provider": provider,
                "device": device,
                "document": document,
                "metric": metric,
                "value": "" if value is None else value,
                "unit": unit,
                "status": status,
            }
        )

    for provider_key, provider_name in (("ocr", "qwen3-vl"),):
        for device in ("cpu", "cuda"):
            result = payload["benchmarks"][provider_key].get(
                device, {"status": STATUS_NOT_VERIFIED}
            )
            status = result.get("status", STATUS_NOT_VERIFIED)
            add(
                provider_name,
                device,
                "all",
                "model_loading_time",
                result.get("model_loading_time_ms"),
                "ms",
                status,
            )
            add(
                provider_name,
                device,
                "png",
                "first_inference_latency",
                result.get("first_inference_latency_ms"),
                "ms",
                status,
            )
            add(
                provider_name,
                device,
                "png",
                "warm_inference_latency_mean",
                result.get("warm_inference_mean_ms"),
                "ms",
                status,
            )
            for document, format_result in result.get("formats", {}).items():
                for metric, unit in (
                    ("processing_time_ms", "ms"),
                    ("pages_per_second", "pages/s"),
                    ("confidence", "probability"),
                    ("gpu_peak_memory_mb", "MiB"),
                ):
                    add(
                        provider_name,
                        device,
                        document,
                        metric,
                        format_result.get(metric),
                        unit,
                        format_result.get("status", status),
                    )
                add(
                    provider_name,
                    device,
                    document,
                    "cpu_rss_delta",
                    format_result.get("cpu_memory", {}).get("rss_delta_mb"),
                    "MiB",
                    format_result.get("status", status),
                )
    postprocessor = payload["benchmarks"]["postprocessor"]
    for stage in postprocessor.get("stages", []):
        add(
            "symspell",
            "cpu",
            "synthetic_text",
            f"{stage['stage']}_latency",
            stage["latency_ms"],
            "ms",
            postprocessor["status"],
        )
        add(
            "symspell",
            "cpu",
            "synthetic_text",
            f"{stage['stage']}_corrections",
            stage["corrections"],
            "count",
            postprocessor["status"],
        )
    return rows


def _validation_markdown(payload: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| {row['requirement']} | {row['status']} | {row['evidence']} |"
        for row in payload["requirements"]
    )
    overall = (
        STATUS_PASS
        if all(row["status"] == STATUS_PASS for row in payload["requirements"])
        else STATUS_NOT_VERIFIED
    )
    runtime = payload["runtime"]
    manifest = payload["model_manifest"]
    screenshot_rows = "\n".join(
        f"- {name}: {item['status']} - {item['evidence']}"
        for name, item in payload["screenshots"].items()
    )
    return f"""# Phase 2 Step 4 OCR Validation Report

Generated: {runtime["timestamp"]}

Overall result: **{overall}**

Step 4 must remain pending unless every requirement is PASS. Boundary tests and decoder
tests do not substitute for a real immutable checkpoint inference.

## Requirement matrix

| Requirement | Result | Evidence |
|---|---|---|
{rows}

## Runtime inventory

- Platform: `{runtime["platform"]}`
- Python: `{runtime["python"].splitlines()[0]}`
- PyTorch: `{runtime["torch"]}`
- Transformers: `{runtime["transformers"]}`
- CUDA available to PyTorch: `{runtime["cuda_available_to_torch"]}`
- CUDA device count: `{runtime["cuda_device_count"]}`
- Physical GPU inventory: `{json.dumps(runtime["physical_gpu"])}`
- Cached Qwen3-VL candidates:
  `{json.dumps(runtime["cached_qwen3_vl_candidates"])}`
- Provider environment present:
  `{json.dumps(runtime["provider_environment_present"], sort_keys=True)}`

## Model manifest

- Status: `{manifest["status"]}`
- Path: `{manifest["path"]}`
- Evidence: {manifest["evidence"]}
- Models: `{json.dumps(manifest["models"], sort_keys=True)}`

## Metrics

The JSON report contains complete per-provider and per-format records. Null or absent
metrics mean the corresponding real provider did not run; no value is inferred.

Confidence distribution: `{json.dumps(payload["confidence_distribution"], sort_keys=True)}`

## Evidence boundaries

- Synthetic, de-identified fixtures are used for decoder and post-processing checks.
- Automated tests may fake only the external checkpoint boundary and are not live-model proof.
- No model is downloaded or selected by this runner.
- OCR error rates require an approved ground-truth corpus; none is inferred from fixture text.

## Screenshot evidence

{screenshot_rows}
"""


def _error_markdown(payload: dict[str, Any]) -> str:
    sections = [
        "# OCR Confusion and Correction Report",
        "",
        "All shown text is synthetic and de-identified.",
        "",
        "## OCR mistakes",
        "",
    ]
    ocr_examples = [
        item for item in payload["error_examples"] if item["category"].startswith("ocr_")
    ]
    if not ocr_examples:
        sections.append(
            "**NOT VERIFIED.** No approved ground-truth corpus and live Qwen3-VL "
            "output were available."
        )
    else:
        for example in ocr_examples:
            sections.append(f"- {example.get('document', 'document')}: {example.get('reason')}")
    labels = {
        "regex": "Regex fixes",
        "medical_abbreviation_dictionary": "Medical abbreviation fixes",
        "symspell": "SymSpell corrections",
    }
    for category, title in labels.items():
        sections.extend(["", f"## {title}", ""])
        examples = [item for item in payload["error_examples"] if item["category"] == category]
        if not examples:
            sections.append("No correction evidence was produced.")
            continue
        for example in examples:
            sections.extend(
                [
                    f"- Corrections: {example['corrections']}; latency: {example['latency_ms']} ms",
                    f"  - Before: `{example['before']}`",
                    f"  - After: `{example['after']}`",
                ]
            )
    return "\n".join(sections) + "\n"


async def run(args: argparse.Namespace) -> dict[str, Any]:
    handler = EvidenceLogHandler()
    provider_logger = logging.getLogger("app.ocr")
    provider_logger.addHandler(handler)
    provider_logger.setLevel(logging.INFO)
    registry = ProviderRegistry()
    register_builtin_providers(registry)
    documents = _synthetic_documents()
    requirements = _requirement_rows()
    error_examples: list[dict[str, Any]] = []
    ocr_results: dict[str, Any] = {}
    try:
        postprocessor = await _validate_postprocessor(registry, error_examples)
        for device in args.devices:
            if device == "cuda" and not torch.cuda.is_available():
                reason = "Installed PyTorch runtime does not expose CUDA."
                ocr_results[device] = {"status": STATUS_NOT_VERIFIED, "reason": reason}
                continue
            ocr_results[device] = await _benchmark_ocr(
                registry, documents, device, args.warm_runs, error_examples
            )
    finally:
        provider_logger.removeHandler(handler)
    payload = {
        "schema_version": "ocr-validation-v1",
        "model_manifest": _validate_model_manifest(),
        "runtime": _runtime_inventory(),
        "requirements": requirements,
        "format_decoding": _validate_decoding(documents),
        "benchmarks": {
            "ocr": ocr_results,
            "postprocessor": postprocessor,
        },
        "confidence_distribution": {},
        "error_examples": error_examples,
        "structured_logs": handler.records,
        "screenshots": _screenshot_inventory(args.output_directory.resolve()),
    }
    payload["confidence_distribution"] = _confidence_distribution(payload)
    _apply_requirement_evidence(payload)
    _write_synthetic_fixtures(documents, args.output_directory.resolve())
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices",
        default="cpu,cuda",
        help="Comma-separated devices to validate (default: cpu,cuda).",
    )
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--output-directory", type=Path, default=REPORTS_DIRECTORY)
    args = parser.parse_args()
    args.devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    if args.warm_runs < 1:
        parser.error("--warm-runs must be at least 1")
    if not args.devices or any(device not in {"cpu", "cuda"} for device in args.devices):
        parser.error("--devices must contain only cpu and/or cuda")
    payload = asyncio.run(run(args))
    _write_reports(payload, args.output_directory.resolve())
    print(json.dumps({"reports": str(args.output_directory.resolve())}, indent=2))


if __name__ == "__main__":
    main()

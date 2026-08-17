"""Controlled benchmark for the Qwen3-VL OCR output-contract experiment.

This module is deliberately outside the FastAPI application.  It reuses the loaded
production provider runtime while switching only the text-generation contract, allowing
the benchmark to compare model-emitted JSON with a deterministic application envelope.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from time import monotonic, perf_counter
from typing import Any
from uuid import uuid4

from app.ocr.providers.contracts import ProviderDocument
from app.ocr.providers.documents import decode_document_pages
from app.ocr.providers.implementations import (
    Qwen3VLOCRProvider,
    _aggregate_document_type,
    _parse_qwen_response,
)
from benchmarks.clinical_performance.pdf_deid import PdfDeidRecord, phi_recovery

_TRANSCRIPTION_PROMPT = (
    "Transcribe all visible text exactly in reading order. Return only the transcription."
)
_OBSERVATIONAL_TERMS = ("metformin", "hemoglobin", "glucose", "hba1c", "cbc", "mri")
_NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?(?:\s*(?:mg/dl|mg|ml|%|mmhg|kg|mcg))?\b", re.I)


@dataclass(frozen=True, slots=True)
class ContractGeneration:
    """Raw result of one page-level generation contract."""

    contract: str
    raw_output: str
    text: str
    document_type: str
    structured_output_valid: bool
    confidence: float | None
    confidence_method: str | None
    generated_tokens: int
    prompt_tokens: int
    generation_latency_ms: float
    total_latency_ms: float
    gpu_memory_allocated_mb: float | None
    gpu_peak_memory_mb: float | None
    warnings: tuple[str, ...]


class _GenerationDeadline:
    """Transformers stopping criterion mirroring the configured provider timeout."""

    def __init__(self, timeout_seconds: float) -> None:
        self._deadline = monotonic() + timeout_seconds
        self.timed_out = False

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
        del input_ids, scores, kwargs
        self.timed_out = monotonic() >= self._deadline
        return self.timed_out


def run_contract_generation(
    provider: Qwen3VLOCRProvider,
    document: ProviderDocument,
    *,
    transcription_only: bool,
) -> ContractGeneration:
    """Generate OCR text with the already initialized provider model and tokenizer.

    It deliberately uses the same decoding arguments, processor, page rendering, device
    transfer, confidence calculation, and generation ceiling as the provider.  Only the
    output prompt and deterministic parsing/envelope boundary differ.
    """

    provider._require_ready()
    started = perf_counter()
    pages = decode_document_pages(
        document,
        pdf_render_dpi=provider._integer("pdf_render_dpi", minimum=72, maximum=600),
        max_image_size=provider._integer("max_image_size", minimum=256),
        max_pages=provider._integer("max_pages", minimum=1),
    )
    provider._reset_gpu_peak()
    prompt = _TRANSCRIPTION_PROMPT if transcription_only else provider._structured_prompt()
    page_results = [
        _run_page(provider, page.image, prompt, transcription_only=transcription_only)
        for page in pages
    ]
    warnings: list[str] = []
    warnings.extend(warning for page_result in page_results for warning in page_result["warnings"])
    text = "\n\n".join(page_result["text"] for page_result in page_results if page_result["text"])
    raw_output = "\n\n".join(page_result["raw_output"] for page_result in page_results)
    document_types = [page_result["document_type"] for page_result in page_results]
    document_type = "unknown" if transcription_only else _aggregate_document_type(document_types)
    structured_output_valid = all(
        page_result["structured_output_valid"] for page_result in page_results
    )
    confidence_values = [page_result["confidence"] for page_result in page_results]
    confidence = (
        sum(value for value in confidence_values if value is not None)
        / sum(value is not None for value in confidence_values)
        if any(value is not None for value in confidence_values)
        else None
    )
    confidence_method = (
        "mean_transcription_token_probability"
        if transcription_only and confidence is not None
        else "mean_generated_token_probability"
        if confidence is not None
        else None
    )
    if not text:
        warnings.append("Model did not return a complete OCR transcription.")
    allocated, peak = provider._gpu_memory()
    return ContractGeneration(
        contract="transcription_only" if transcription_only else "structured_json",
        raw_output=raw_output,
        text=text,
        document_type=document_type,
        structured_output_valid=structured_output_valid,
        confidence=confidence,
        confidence_method=confidence_method,
        generated_tokens=sum(page_result["generated_tokens"] for page_result in page_results),
        prompt_tokens=sum(page_result["prompt_tokens"] for page_result in page_results),
        generation_latency_ms=round(
            sum(page_result["generation_latency_ms"] for page_result in page_results), 3
        ),
        total_latency_ms=round((perf_counter() - started) * 1000, 3),
        gpu_memory_allocated_mb=allocated,
        gpu_peak_memory_mb=peak,
        warnings=tuple(warnings),
    )


def _run_page(
    provider: Qwen3VLOCRProvider,
    image: Any,
    prompt: str,
    *,
    transcription_only: bool,
) -> dict[str, Any]:
    """Run one rendered PDF page independently, matching provider page semantics."""

    messages = [
        [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
            }
        ]
    ]
    inputs = provider._processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    inputs = provider._move_inputs(inputs)
    input_length = int(inputs["input_ids"].shape[1])
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": provider._integer("max_new_tokens", minimum=1),
        "do_sample": provider._boolean("do_sample"),
        "num_beams": provider._integer("num_beams", minimum=1),
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if generate_kwargs["do_sample"]:
        generate_kwargs["temperature"] = provider._float("temperature", minimum=0.0001)
        generate_kwargs["top_p"] = provider._float("top_p", minimum=0.0001, maximum=1)
    provider._torch.manual_seed(provider._integer("seed", minimum=0))
    if provider._resolved_device.startswith("cuda"):
        provider._torch.cuda.manual_seed_all(provider._integer("seed", minimum=0))
    from transformers import StoppingCriteriaList

    deadline = _GenerationDeadline(provider._float("timeout_seconds", minimum=0.001))
    generation_started = perf_counter()
    with provider._inference_lock, provider._torch.inference_mode():
        output = provider._model.generate(
            **inputs,
            **generate_kwargs,
            stopping_criteria=StoppingCriteriaList([deadline]),
        )
    generation_latency_ms = round((perf_counter() - generation_started) * 1000, 3)
    generated = output.sequences[:, input_length:]
    raw_output = provider._processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    confidences, token_counts = provider._generated_confidences(generated, output.scores)
    warnings: list[str] = []
    if deadline.timed_out:
        warnings.append("Generation reached the configured 300-second timeout.")
    if transcription_only:
        text, document_type, structured_output_valid = raw_output, "unknown", False
    else:
        try:
            parsed = _parse_qwen_response(raw_output)
            text, document_type, structured_output_valid = (
                parsed["text"],
                parsed["document_type"],
                True,
            )
        except Exception as exc:
            text, document_type, structured_output_valid = "", "unknown", False
            warnings.append(f"Structured response could not be parsed: {type(exc).__name__}.")
    return {
        "raw_output": raw_output,
        "text": text,
        "document_type": document_type,
        "structured_output_valid": structured_output_valid,
        "confidence": confidences[0],
        "generated_tokens": token_counts[0],
        "prompt_tokens": input_length,
        "generation_latency_ms": generation_latency_ms,
        "warnings": warnings,
    }


def deterministic_envelope(
    generation: ContractGeneration,
    *,
    normalized_text: str | None,
    page_count: int,
    provider: Qwen3VLOCRProvider,
    postprocessing_warning: str | None = None,
) -> dict[str, Any]:
    """Construct the existing OCR-result information without model-generated keys."""

    metadata = provider.metadata()
    configuration = metadata.configuration
    warnings = list(generation.warnings)
    if postprocessing_warning:
        warnings.append(postprocessing_warning)
    return {
        "request_id": str(uuid4()),
        "report_id": str(uuid4()),
        "ocr_id": str(uuid4()),
        "provider": metadata.provider_name,
        "provider_version": metadata.provider_version,
        "pipeline_version": "ocr-output-contract-experiment-v1",
        "raw_text": generation.text,
        "normalized_text": normalized_text,
        "processing_time_ms": generation.total_latency_ms,
        "page_count": page_count,
        "status": "completed" if generation.text else "failed",
        "warnings": warnings,
        "metadata": {
            "model_name": configuration.get("model_name"),
            "model_revision": configuration.get("model_revision"),
            "document_type": generation.document_type,
            "document_type_source": "benchmark_unknown"
            if generation.document_type == "unknown"
            else "qwen3-vl_prompt",
            "confidence": generation.confidence,
            "confidence_method": generation.confidence_method,
            "generated_tokens": generation.generated_tokens,
        },
    }


def compare_generations(
    record: PdfDeidRecord,
    baseline: ContractGeneration,
    candidate: ContractGeneration,
) -> dict[str, Any]:
    """Compute supported and explicitly observational contract-comparison evidence."""

    baseline_phi = phi_recovery(record.phi_tokens, baseline.text)
    candidate_phi = phi_recovery(record.phi_tokens, candidate.text)
    baseline_numbers = sorted(set(_NUMBER_PATTERN.findall(baseline.text)))
    candidate_numbers = sorted(set(_NUMBER_PATTERN.findall(candidate.text)))
    missing_numbers = [value for value in baseline_numbers if value not in candidate_numbers]
    observable_terms = {
        term: {
            "baseline": term in baseline.text.casefold(),
            "candidate": term in candidate.text.casefold(),
        }
        for term in _OBSERVATIONAL_TERMS
    }
    return {
        "filename": record.filename,
        "difficulty": record.difficulty,
        "baseline": _generation_payload(baseline, baseline_phi),
        "candidate": _generation_payload(candidate, candidate_phi),
        "speedup": round(baseline.total_latency_ms / candidate.total_latency_ms, 4)
        if candidate.total_latency_ms > 0
        else None,
        "text_similarity_observational": round(
            SequenceMatcher(a=baseline.text, b=candidate.text).ratio(), 6
        ),
        "numbers_observational": {
            "baseline_numbers": baseline_numbers,
            "candidate_numbers": candidate_numbers,
            "missing_from_candidate": missing_numbers,
            "preserved": not missing_numbers,
        },
        "medical_terms_observational": observable_terms,
        "cer": "NOT_VERIFIED",
        "wer": "NOT_VERIFIED",
    }


def _generation_payload(generation: ContractGeneration, phi: dict[str, object]) -> dict[str, Any]:
    return {
        "total_latency_ms": generation.total_latency_ms,
        "generation_latency_ms": generation.generation_latency_ms,
        "generated_tokens": generation.generated_tokens,
        "tokens_per_second": round(
            generation.generated_tokens / max(generation.generation_latency_ms / 1000, 1e-9), 6
        ),
        "structured_output_valid": generation.structured_output_valid,
        "document_type": generation.document_type,
        "confidence": generation.confidence,
        "confidence_method": generation.confidence_method,
        "phi_recovery": phi,
        "gpu_memory_allocated_mb": generation.gpu_memory_allocated_mb,
        "gpu_peak_memory_mb": generation.gpu_peak_memory_mb,
        "warnings": list(generation.warnings),
    }


def write_contract_artifacts(payload: dict[str, Any], output_directory) -> None:
    """Write reproducible JSON, CSV, and concise Markdown benchmark evidence."""

    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "ocr_output_contract_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows = [
        "filename,difficulty,baseline_ms,candidate_ms,speedup,baseline_tokens,candidate_tokens,baseline_phi_recovery,candidate_phi_recovery"
    ]
    for item in payload["comparisons"]:
        baseline = item["baseline"]
        candidate = item["candidate"]
        rows.append(
            ",".join(
                str(value)
                for value in (
                    item["filename"],
                    item["difficulty"],
                    baseline["total_latency_ms"],
                    candidate["total_latency_ms"],
                    item["speedup"],
                    baseline["generated_tokens"],
                    candidate["generated_tokens"],
                    baseline["phi_recovery"]["recovery_rate"],
                    candidate["phi_recovery"]["recovery_rate"],
                )
            )
        )
    (output_directory / "ocr_output_contract_metrics.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    lines = [
        "# OCR Output Contract Optimization Experiment",
        "",
        f"Status: {payload['status']}",
        "",
        "| Document | Baseline ms | Candidate ms | Speedup | Baseline PHI | Candidate PHI |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["comparisons"]:
        baseline = item["baseline"]
        candidate = item["candidate"]
        lines.append(
            f"| {item['filename']} | {baseline['total_latency_ms']} | "
            f"{candidate['total_latency_ms']} | "
            f"{item['speedup']} | {baseline['phi_recovery']['recovery_rate']} | "
            f"{candidate['phi_recovery']['recovery_rate']} |"
        )
    lines.extend(("", "## Decision", "", payload["recommendation"], ""))
    (output_directory / "ocr_output_contract_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

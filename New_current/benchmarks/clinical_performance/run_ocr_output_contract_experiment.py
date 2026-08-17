"""Run the isolated structured-JSON versus transcription-only OCR experiment offline."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ocr.providers.contracts import ProviderDocument
from app.ocr.providers.implementations import Qwen3VLOCRProvider, SymSpellPostProcessor
from benchmarks.clinical_performance.ocr_output_contract import (
    compare_generations,
    deterministic_envelope,
    run_contract_generation,
    write_contract_artifacts,
)
from benchmarks.clinical_performance.pdf_deid import load_pdf_deid_dataset
from benchmarks.ocr.run_validation import _benchmark_postprocessor_configuration

_PRODUCTION_PROMPT = "Transcribe the medical document exactly, preserving reading order."


def _configuration(
    repository_root: Path, *, max_image_size: int, pdf_render_dpi: int
) -> dict[str, str]:
    """Return the pinned BF16 configuration without reading or changing deployment defaults."""

    return {
        "provider_version": "benchmark-output-contract-v1",
        "model_name": "Qwen/Qwen3-VL-4B-Instruct",
        "model_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        "hf_cache_dir": str(repository_root / ".model-cache" / "qwen3-vl"),
        "local_files_only": "true",
        "device": "auto",
        "allow_cpu_fallback": "true",
        "dtype": "bfloat16",
        "batch_size": "1",
        "max_image_size": str(max_image_size),
        "max_pages": "50",
        "pdf_render_dpi": str(pdf_render_dpi),
        "timeout_seconds": "300",
        "max_new_tokens": "2048",
        "do_sample": "false",
        "temperature": "0.1",
        "top_p": "1.0",
        "num_beams": "1",
        "prompt": _PRODUCTION_PROMPT,
        "prompt_version": "deployment-approved-v1",
        "confidence_threshold": "0.60",
        "confidence_calibration_version": "token-probability-uncalibrated-v1",
        "seed": "0",
    }


def _normalize(
    postprocessor: SymSpellPostProcessor, text: str, document_type: str
) -> tuple[str | None, str | None]:
    if not text:
        return None, "Post-processing was skipped because OCR did not return complete text."
    try:
        return postprocessor.normalize(text, document_type=document_type).normalized_text, None
    except Exception as exc:
        return None, f"Post-processing did not run: {type(exc).__name__}."


async def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    records = load_pdf_deid_dataset(arguments.dataset)
    selected = [
        *[record for record in records if record.difficulty == "medium"][:2],
        *[record for record in records if record.difficulty == "hard"][:2],
    ]
    if len(selected) != 4 or sum(record.difficulty == "medium" for record in selected) != 2:
        raise ValueError("Dataset must contain at least two Medium and two Hard PDFs.")

    provider = Qwen3VLOCRProvider(
        _configuration(
            arguments.repository_root,
            max_image_size=arguments.max_image_size,
            pdf_render_dpi=arguments.pdf_render_dpi,
        )
    )
    postprocessor = SymSpellPostProcessor(_benchmark_postprocessor_configuration())
    await provider.initialize()
    await postprocessor.initialize()
    comparisons = []
    envelopes = []
    try:
        for record in selected:
            document = ProviderDocument(
                content=record.path.read_bytes(), file_type="pdf", filename=record.filename
            )
            baseline = run_contract_generation(provider, document, transcription_only=False)
            candidate = run_contract_generation(provider, document, transcription_only=True)
            baseline_normalized, baseline_post_warning = _normalize(
                postprocessor, baseline.text, baseline.document_type
            )
            candidate_normalized, candidate_post_warning = _normalize(
                postprocessor, candidate.text, candidate.document_type
            )
            comparisons.append(compare_generations(record, baseline, candidate))
            envelopes.append(
                {
                    "filename": record.filename,
                    "baseline": deterministic_envelope(
                        baseline,
                        normalized_text=baseline_normalized,
                        page_count=1,
                        provider=provider,
                        postprocessing_warning=baseline_post_warning,
                    ),
                    "candidate": deterministic_envelope(
                        candidate,
                        normalized_text=candidate_normalized,
                        page_count=1,
                        provider=provider,
                        postprocessing_warning=candidate_post_warning,
                    ),
                }
            )
    finally:
        await postprocessor.shutdown()
        await provider.shutdown()

    complete = all(
        item["baseline"]["structured_output_valid"]
        and item["candidate"]["phi_recovery"]["recovery_rate"] is not None
        for item in comparisons
    )
    candidate_complete = all(not item["candidate"]["warnings"] for item in comparisons)
    candidate_improved = all((item["speedup"] or 0) > 1 for item in comparisons)
    recommendation = (
        "MORE_VALIDATION_REQUIRED: controlled candidate evidence is complete, but synthetic "
        "PHI-token mappings cannot establish clinical transcription fidelity."
        if complete and candidate_complete and candidate_improved
        else "REJECT: the candidate did not meet the controlled benchmark promotion criteria."
    )
    return {
        "status": "COMPLETED",
        "scope": "benchmark-only; production API and configuration unchanged",
        "configuration": {
            "model": "Qwen/Qwen3-VL-4B-Instruct",
            "revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
            "dtype": "bfloat16",
            "device": "auto",
            "max_image_size": arguments.max_image_size,
            "pdf_render_dpi": arguments.pdf_render_dpi,
            "max_new_tokens": 2048,
            "timeout_seconds": 300,
            "postprocessing": (
                "same synthetic benchmark fixture for baseline and candidate; "
                "production mount not available locally"
            ),
        },
        "document_type_options": {
            "A": (
                "Candidate implemented: unknown; it is fail-safe and causes "
                "review-required behavior."
            ),
            "B": (
                "NOT TESTED: a second model generation would add serial latency "
                "and is not the lowest-cost option."
            ),
            "C": (
                "Applicable only to the existing native digital-PDF path; scanned "
                "PDFs remain unknown."
            ),
            "recommendation": (
                "A for scanned/image OCR until a separately validated classification "
                "policy is approved; retain the native digital-PDF classification path."
            ),
        },
        "confidence": {
            "baseline": "mean_generated_token_probability includes structured envelope tokens.",
            "candidate": (
                "mean_transcription_token_probability uses transcription tokens only; "
                "values are not directly comparable."
            ),
        },
        "comparisons": comparisons,
        "deterministic_envelopes": envelopes,
        "recommendation": recommendation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--max-image-size", type=int, default=512)
    parser.add_argument("--pdf-render-dpi", type=int, default=96)
    arguments = parser.parse_args()
    payload = asyncio.run(_run(arguments))
    write_contract_artifacts(payload, arguments.output)


if __name__ == "__main__":
    main()

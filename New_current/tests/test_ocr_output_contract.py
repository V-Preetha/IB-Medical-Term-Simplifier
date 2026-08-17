"""Tests for the isolated OCR output-contract benchmark adapter."""

from __future__ import annotations

from pathlib import Path

from benchmarks.clinical_performance.ocr_output_contract import (
    ContractGeneration,
    compare_generations,
)
from benchmarks.clinical_performance.pdf_deid import PdfDeidRecord


def _generation(*, contract: str, text: str, latency: float) -> ContractGeneration:
    return ContractGeneration(
        contract=contract,
        raw_output=text,
        text=text,
        document_type="scanned_pdf" if contract == "structured_json" else "unknown",
        structured_output_valid=contract == "structured_json",
        confidence=0.9,
        confidence_method="mean_generated_token_probability",
        generated_tokens=10,
        prompt_tokens=20,
        generation_latency_ms=latency,
        total_latency_ms=latency,
        gpu_memory_allocated_mb=1.0,
        gpu_peak_memory_mb=2.0,
        warnings=(),
    )


def test_candidate_comparison_keeps_phi_and_numbers_as_supported_metrics() -> None:
    record = PdfDeidRecord(
        filename="synthetic.pdf",
        difficulty="medium",
        path=Path("synthetic.pdf"),
        phi_tokens=("Alice", "01/02/2020"),
    )
    baseline = _generation(
        contract="structured_json",
        text="Alice received Metformin 10 mg on 01/02/2020.",
        latency=100,
    )
    candidate = _generation(
        contract="transcription_only",
        text="Alice received Metformin 10 mg on 01/02/2020.",
        latency=50,
    )

    comparison = compare_generations(record, baseline, candidate)

    assert comparison["speedup"] == 2.0
    assert comparison["baseline"]["phi_recovery"]["recovery_rate"] == 1.0
    assert comparison["candidate"]["phi_recovery"]["recovery_rate"] == 1.0
    assert comparison["numbers_observational"]["preserved"] is True
    assert comparison["cer"] == "NOT_VERIFIED"
    assert comparison["wer"] == "NOT_VERIFIED"


def test_candidate_comparison_reports_missing_baseline_number() -> None:
    record = PdfDeidRecord("synthetic.pdf", "hard", Path("synthetic.pdf"), ())
    baseline = _generation(contract="structured_json", text="Glucose 125 mg/dL", latency=100)
    candidate = _generation(contract="transcription_only", text="Glucose value", latency=50)

    comparison = compare_generations(record, baseline, candidate)

    assert comparison["numbers_observational"]["preserved"] is False
    assert comparison["numbers_observational"]["missing_from_candidate"] == ["125 mg/dL"]

# ruff: noqa: E501
import json
import subprocess
import sys

from benchmarks.clinical_performance.harness import (
    BenchmarkDocument,
    error_rate,
    evaluate_ocr,
    evaluate_simplification,
    recommendation,
    write_artifacts,
)


def test_ocr_metrics_remain_unmeasured_without_reviewed_gold() -> None:
    document = BenchmarkDocument("lab-1", "lab_report", "approved/lab-1.png")
    result = evaluate_ocr(document, {"text": "HbA1c 7.2%", "latency_ms": 10})
    assert result["cer"] is None
    assert result["wer"] is None


def test_ocr_metrics_and_protected_content_are_deterministic() -> None:
    document = BenchmarkDocument(
        "lab-1", "lab_report", "approved/lab-1.png", ocr_gold_text="HbA1c 7.2% no pneumonia"
    )
    result = evaluate_ocr(document, {"text": "HbA1c 7.2% no pneumonia"})
    assert result["cer"] == 0.0
    assert result["wer"] == 0.0
    assert all(result["protected_preservation"].values())
    assert error_rate("one two", "one", characters=False) == 0.5


def test_simplification_completeness_and_fail_closed_recommendation(tmp_path) -> None:
    output = {
        "levels": {
            name: {"simplified_report": "Metformin 500 mg twice daily. No pneumonia."}
            for name in ("clinical", "general_public", "child_friendly")
        },
        "generated_tokens": 20,
        "schema_valid": True,
        "verification_verdict": "PASS",
    }
    result = evaluate_simplification("Metformin 500 mg twice daily. No pneumonia.", output)
    assert result["readability_level_complete"] is True
    assert recommendation(safety_passed=False, quality_complete=True, speedup=2.0) == "REJECT"
    assert recommendation(safety_passed=True, quality_complete=False, speedup=2.0) == "MORE_VALIDATION_REQUIRED"
    summary = {"status": "NOT_RUN", "comparisons": [{"candidate": "ocr_512px", "recommendation": "MORE_VALIDATION_REQUIRED"}]}
    write_artifacts(summary, tmp_path)
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "NOT_RUN"
    assert (tmp_path / "summary.csv").is_file()
    assert (tmp_path / "summary.md").is_file()


def test_runner_writes_pending_candidate_artifacts(tmp_path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps({"document_id": "small-1", "category": "small_text_report", "document_path": "approved/small.png"}) + "\n")
    output = tmp_path / "artifacts"
    subprocess.run(
        [sys.executable, "benchmarks/clinical_performance/run_harness.py", "--dataset", str(dataset), "--output", str(output)],
        check=True,
    )
    assert json.loads((output / "summary.json").read_text())["document_count"] == 1

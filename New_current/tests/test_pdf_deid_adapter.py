from pathlib import Path

from benchmarks.clinical_performance.pdf_deid import (
    DATASET_ID,
    load_pdf_deid_dataset,
    phi_recovery,
)


def test_pdf_deid_adapter_discovers_synthetic_corpus() -> None:
    root = Path("dataset/pdf-deid-dataset-main")
    records = load_pdf_deid_dataset(root)
    assert DATASET_ID == "pdf_deid_synthetic_medical_v1"
    assert {record.difficulty for record in records} == {"easy", "medium", "hard"}
    assert len(records) == 50
    assert sum(record.difficulty == "easy" for record in records) == 30
    assert sum(record.difficulty == "medium" for record in records) == 10
    assert sum(record.difficulty == "hard" for record in records) == 10


def test_phi_recovery_never_claims_full_ocr_metrics() -> None:
    result = phi_recovery(("Alice Smith", "01/01/2000", "HOSP123"), "Alice Smith HOSP123")
    assert result["recovered_unique_tokens"] == 2
    assert result["cer"] == "NOT_VERIFIED"

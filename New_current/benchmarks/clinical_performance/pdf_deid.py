"""Adapter for the synthetic PDF Deid OCR-robustness corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATASET_ID = "pdf_deid_synthetic_medical_v1"


@dataclass(frozen=True, slots=True)
class PdfDeidRecord:
    """One synthetic PDF and its PHI-token-only annotations."""

    filename: str
    difficulty: str
    path: Path
    phi_tokens: tuple[str, ...]


def load_pdf_deid_dataset(root: Path) -> list[PdfDeidRecord]:
    """Discover 30 Easy, 10 Medium, and 10 Hard PDFs with cumulative PHI mappings."""

    mapping_path = root / "Mapping" / "all_phi" / "pdf_deid_gts_hard.json"
    annotations = json.loads(mapping_path.read_text(encoding="utf-8"))
    records = []
    for difficulty in ("Easy", "Medium", "Hard"):
        for path in sorted((root / "PDF_Original" / difficulty).glob("*.pdf")):
            values = annotations.get(path.name)
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                raise ValueError(f"Missing PHI-only mapping for {path.name}.")
            records.append(PdfDeidRecord(path.name, difficulty.casefold(), path, tuple(values)))
    return records


def phi_recovery(tokens: tuple[str, ...], ocr_text: str) -> dict[str, object]:
    """Measure only supported annotation recovery, never CER/WER."""

    unique = tuple(dict.fromkeys(value.strip() for value in tokens if value.strip()))
    folded = ocr_text.casefold()
    recovered = [value for value in unique if value.casefold() in folded]
    return {
        "annotation_type": "synthetic_phi_token_list",
        "annotated_unique_tokens": len(unique),
        "recovered_unique_tokens": len(recovered),
        "recovery_rate": round(len(recovered) / len(unique), 6) if unique else None,
        "missing_tokens": [value for value in unique if value not in recovered],
        "cer": "NOT_VERIFIED",
        "wer": "NOT_VERIFIED",
        "exact_text_agreement": "NOT_VERIFIED",
    }

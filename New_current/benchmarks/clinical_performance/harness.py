# ruff: noqa: E501
"""Model-neutral evaluation helpers for approved de-identified benchmark artifacts."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CATEGORIES = frozenset(
    {
        "lab_report", "prescription", "discharge_summary", "radiology_report",
        "consultation_note", "handwritten_scanned_report", "table_heavy_report",
        "small_text_report", "multi_page_report",
    }
)
PROTECTED = re.compile(
    r"\b(?:\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?\s*(?:%|mg|mcg|g|mL|L|mmHg|mg/dL|g/dL|mmol/L)?|"
    r"twice daily|once daily|three times daily|no |not |without |left |right )\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class BenchmarkDocument:
    """One de-identified input. Gold transcription is optional."""

    document_id: str
    category: str
    document_path: str
    ocr_gold_text: str | None = None
    simplification_source_text: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.document_id or self.category not in CATEGORIES or not self.document_path:
            raise ValueError("Benchmark document has invalid required metadata.")


def load_dataset(path: Path) -> list[BenchmarkDocument]:
    """Load JSONL input records without inventing clinical labels."""

    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(BenchmarkDocument(**json.loads(line)))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid dataset entry at line {number}.") from exc
    return records


def levenshtein(left: list[str], right: list[str]) -> int:
    """Return edit distance without an added runtime package."""

    if len(left) < len(right):
        left, right = right, left
    row = list(range(len(right) + 1))
    for index, token in enumerate(left, 1):
        next_row = [index]
        for right_index, other in enumerate(right, 1):
            next_row.append(
                min(next_row[-1] + 1, row[right_index] + 1, row[right_index - 1] + (token != other))
            )
        row = next_row
    return row[-1]


def error_rate(reference: str | None, candidate: str, *, characters: bool) -> float | None:
    """Return CER or WER, explicitly null when reviewed gold text is absent."""

    if not reference or not reference.strip():
        return None
    expected = list(reference) if characters else reference.split()
    actual = list(candidate) if characters else candidate.split()
    return round(levenshtein(expected, actual) / max(len(expected), 1), 6)


def protected_preservation(source: str, candidate: str) -> dict[str, bool]:
    """Check numbers, units, frequency, negation, and laterality source tokens."""

    tokens = {match.group(0).casefold().strip() for match in PROTECTED.finditer(source)}
    candidate_folded = candidate.casefold()
    return {token: token in candidate_folded for token in sorted(tokens)}


def evaluate_ocr(document: BenchmarkDocument, output: dict[str, Any]) -> dict[str, Any]:
    """Normalize OCR quality, protected content, and runtime evidence."""

    text = str(output.get("text", ""))
    return {
        "document_id": document.document_id,
        "category": document.category,
        "cer": error_rate(document.ocr_gold_text, text, characters=True),
        "wer": error_rate(document.ocr_gold_text, text, characters=False),
        "protected_preservation": protected_preservation(document.ocr_gold_text or "", text) or None,
        "latency_ms": output.get("latency_ms"),
        "gpu_memory_mb": output.get("gpu_memory_mb"),
        "visual_resolution": output.get("visual_resolution"),
        "generated_tokens": output.get("generated_tokens"),
        "table_or_small_text_failure": output.get("table_or_small_text_failure"),
    }


def evaluate_simplification(source: str, output: dict[str, Any]) -> dict[str, Any]:
    """Normalize structured-output, safety, verification, and performance evidence."""

    levels = output.get("levels", {})
    complete = all(isinstance(levels.get(name), dict) and levels[name].get("simplified_report") for name in ("clinical", "general_public", "child_friendly"))
    rendered = " ".join(str(level.get("simplified_report", "")) for level in levels.values() if isinstance(level, dict))
    return {
        "schema_valid": bool(output.get("schema_valid", complete)),
        "readability_level_complete": complete,
        "generated_tokens": output.get("generated_tokens"),
        "prompt_tokens": output.get("prompt_tokens"),
        "generation_latency_ms": output.get("generation_latency_ms"),
        "tokens_per_second": output.get("tokens_per_second"),
        "grounding": protected_preservation(source, rendered),
        "unsupported_claims": output.get("unsupported_claims"),
        "verification_verdict": output.get("verification_verdict"),
        "verification_confidence": output.get("verification_confidence"),
    }


def recommendation(*, safety_passed: bool | None, quality_complete: bool, speedup: float | None) -> str:
    """Reject safety failures; never auto-promote incomplete evidence."""

    if safety_passed is False:
        return "REJECT"
    if not quality_complete or speedup is None:
        return "MORE_VALIDATION_REQUIRED"
    return "PROMOTE"


def write_artifacts(summary: dict[str, Any], destination: Path) -> None:
    """Write machine-readable JSON, CSV, and Markdown comparison artifacts."""

    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    rows = summary.get("comparisons", [])
    keys = sorted({key for row in rows for key in row}) or ["candidate", "recommendation"]
    with (destination / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Clinical Performance Benchmark", "", f"Status: {summary.get('status', 'NOT_RUN')}", "", "| Candidate | Recommendation |", "| --- | --- |"]
    lines.extend(f"| {row.get('candidate', 'unknown')} | {row.get('recommendation', 'MORE_VALIDATION_REQUIRED')} |" for row in rows)
    (destination / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def document_to_json(document: BenchmarkDocument) -> dict[str, Any]:
    """Return the dataset record in its stable JSON representation."""

    return asdict(document)

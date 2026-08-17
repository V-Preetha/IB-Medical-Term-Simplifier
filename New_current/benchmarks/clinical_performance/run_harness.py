"""Create reproducible pending benchmark artifacts from an approved dataset manifest."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.clinical_performance.harness import load_dataset, write_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    documents = load_dataset(arguments.dataset)
    categories = Counter(document.category for document in documents)
    write_artifacts(
        {
            "status": "READY_FOR_APPROVED_RUN_ARTIFACTS",
            "dataset": str(arguments.dataset),
            "document_count": len(documents),
            "category_counts": dict(sorted(categories.items())),
            "comparisons": [
                {"candidate": value, "recommendation": "MORE_VALIDATION_REQUIRED"}
                for value in (
                    "ocr_512px",
                    "fp16",
                    "int8",
                    "4bit",
                    "compact_simplification_schema",
                    "future_model_replacement",
                )
            ],
        },
        arguments.output,
    )


if __name__ == "__main__":
    main()

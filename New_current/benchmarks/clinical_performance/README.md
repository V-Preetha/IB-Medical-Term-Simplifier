# Clinical Performance Benchmark Harness

This offline harness compares experimental artifacts against the frozen production baseline.
It does not invoke cloud services, change application configuration, or create clinical labels.

Place approved de-identified documents in dataset.jsonl using dataset.schema.json. Gold OCR
text is optional: CER and WER remain null until an independently reviewed transcription is
provided. The schema supports lab reports, prescriptions, discharge summaries, radiology
reports, consultation notes, handwritten/scanned reports, table-heavy reports, small-text
reports, and multi-page reports.

The model-neutral helpers normalize baseline and candidate artifacts for 512px OCR, BF16,
FP16, INT8, 4-bit, compact-schema, and future-model experiments. They record latency, RAM/
VRAM supplied by the runner, tokens, protected source-token preservation, JSON completeness,
grounding, and Verification verdicts. Recommendation is PROMOTE only when complete quality
evidence and a safety pass are supplied; otherwise it remains MORE_VALIDATION_REQUIRED.

write_artifacts emits summary.json, summary.csv, and summary.md. Current measured production
latencies are captured in baseline/current_production_baseline.json and are not clinical-
quality evidence.

Validate an approved dataset and create empty, reproducible comparison artifacts:

    python benchmarks/clinical_performance/run_harness.py --dataset benchmarks/clinical_performance/dataset.jsonl --output benchmarks/clinical_performance/artifacts/run-name

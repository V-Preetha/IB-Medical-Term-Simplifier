# Medical NER Candidate Benchmark

Generated: 2026-08-04T18:17:46.722508+00:00

Dataset: `E:\Internships\Icebrkr\Medical-Term-Simplifier\New_current\benchmarks\ner\dataset_template.jsonl` (4 synthetic de-identified records)  
CUDA available: `False`  
Approved winner: **None**

## Recommendation

**biomedical-ner-all** is recommended for explicit production-model review. It has not been
approved or integrated. Basis: Highest macro F1 across all eight canonical entity types; overall F1 and mean inference latency are deterministic tie-breakers.

## Leaderboard

| Rank | Candidate | Macro F1 | Overall F1 | Mean latency ms |
| ---: | --- | ---: | ---: | ---: |
| 1 | biomedical-ner-all | 0.381250 | 0.333333 | 22.654 |
| 2 | modernbert-biomedical-ner | 0.170833 | 0.222222 | 56.289 |
| 3 | openmed-gliner | 0.083333 | 0.086957 | 114.428 |

## Overall metrics

| Candidate | Status | Precision | Recall | F1 | Accuracy | FP | FN | Load ms | Latency ms | RAM MiB | GPU MiB | Tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| openmed-gliner | PASS | 0.125000 | 0.066667 | 0.086957 | 0.045455 | 7 | 14 | 6444.307 | 114.428 | 1972.750 | NOT VERIFIED | 165.114 |
| biomedical-ner-all | PASS | 0.333333 | 0.333333 | 0.333333 | 0.200000 | 10 | 10 | 125.386 | 22.654 | 1217.066 | NOT VERIFIED | 1257.021 |
| modernbert-biomedical-ner | PASS | 0.250000 | 0.200000 | 0.222222 | 0.125000 | 9 | 12 | 175.454 | 56.289 | 869.730 | NOT VERIFIED | 386.047 |

## Per-entity metrics

### openmed-gliner

| Entity type | Precision | Recall | F1 | Accuracy | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Disease | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 3 |
| Symptom | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 1 |
| Medication | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1 | 1 |
| Procedure | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 2 |
| Anatomy | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1 | 2 |
| Laboratory Test | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 2 | 1 |
| Measurement | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3 | 3 |
| Medical Abbreviation | 1.000000 | 0.500000 | 0.666667 | 0.500000 | 0 | 1 |
### biomedical-ner-all

| Entity type | Precision | Recall | F1 | Accuracy | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Disease | 1.000000 | 0.666667 | 0.800000 | 0.666667 | 0 | 1 |
| Symptom | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 |
| Medication | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 |
| Procedure | 0.166667 | 0.500000 | 0.250000 | 0.142857 | 5 | 1 |
| Anatomy | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3 | 2 |
| Laboratory Test | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 1 |
| Measurement | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 2 | 3 |
| Medical Abbreviation | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 2 |
### modernbert-biomedical-ner

| Entity type | Precision | Recall | F1 | Accuracy | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Disease | 0.142857 | 0.333333 | 0.200000 | 0.111111 | 6 | 2 |
| Symptom | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 1 |
| Medication | 0.500000 | 1.000000 | 0.666667 | 0.500000 | 1 | 0 |
| Procedure | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 2 |
| Anatomy | 0.500000 | 0.500000 | 0.500000 | 0.333333 | 1 | 1 |
| Laboratory Test | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1 | 1 |
| Measurement | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 3 |
| Medical Abbreviation | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 2 |

## Evidence limits

- Every candidate used the same synthetic, de-identified normalized OCR text and
  exact-span references.
- GPU memory is NOT VERIFIED when CUDA is unavailable; it is never reported as zero.
- This benchmark is evidence for a recommendation, not production approval or integration.

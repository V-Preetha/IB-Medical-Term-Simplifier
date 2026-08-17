# Phase 2 Step 4 OCR Validation Report

Generated: 2026-08-08T19:13:39.583609+00:00

Overall result: **NOT VERIFIED**

Step 4 must remain pending unless every requirement is PASS. Boundary tests and decoder
tests do not substitute for a real immutable checkpoint inference.

## Requirement matrix

| Requirement | Result | Evidence |
|---|---|---|
| Model manifest verification | PASS | Every manifest model, immutable revision, provider configuration, and local snapshot was verified. |
| Qwen3-VL loads correctly | FAIL | The PDF exceeds the configured 1-page limit. |
| OCR runs on CPU | FAIL | The PDF exceeds the configured 1-page limit. |
| OCR runs on CUDA (if available) | NOT VERIFIED | did not run |
| PDF | NOT VERIFIED | Decoder PASS; live Qwen3-VL The PDF exceeds the configured 1-page limit. |
| Multi-page PDF | NOT VERIFIED | Decoder PASS; live Qwen3-VL The PDF exceeds the configured 1-page limit. |
| PNG | NOT VERIFIED | Decoder PASS; live Qwen3-VL The PDF exceeds the configured 1-page limit. |
| JPEG | NOT VERIFIED | Decoder PASS; live Qwen3-VL The PDF exceeds the configured 1-page limit. |
| TIFF | NOT VERIFIED | Decoder PASS; live Qwen3-VL The PDF exceeds the configured 1-page limit. |
| Confidence generation | NOT VERIFIED | No live model confidence values were produced. |
| Processing time | NOT VERIFIED | No live model inference timings were produced. |
| OCR postprocessing | PASS | Synthetic pipeline completed with 13 corrections in 1.715 ms. |
| Structured logging | PASS | Captured 3 provider records with structured lifecycle/stage fields. |

## Runtime inventory

- Platform: `Windows-11-10.0.26200-SP0`
- Python: `3.12.13 (main, Mar  3 2026, 15:01:35) [MSC v.1944 64 bit (AMD64)]`
- PyTorch: `2.13.0+cpu`
- Transformers: `4.57.6`
- CUDA available to PyTorch: `False`
- CUDA device count: `0`
- Physical GPU inventory: `["NVIDIA GeForce RTX 5050 Laptop GPU, 8151 MiB, 592.27"]`
- Cached Qwen3-VL candidates:
  `["models--Qwen--Qwen3-VL-4B-Instruct"]`
- Provider environment present:
  `{"ocr": true, "postprocessor": false}`

## Model manifest

- Status: `PASS`
- Path: `E:\Internships\Icebrkr\Medical-Term-Simplifier\MODEL_MANIFEST.md`
- Evidence: Every manifest model, immutable revision, provider configuration, and local snapshot was verified.
- Models: `{"qwen3-vl": {"cache_path": "E:\\Internships\\Icebrkr\\Medical-Term-Simplifier\\New_current\\.model-cache\\qwen3-vl", "checksum_status": "NOT PROVIDED", "expected_sha256": "PENDING_APPROVAL", "license": "Apache-2.0", "observed_sha256": null, "pinned_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17", "provider": "qwen3-vl", "provider_configuration_matches": true, "repository_id": "Qwen/Qwen3-VL-4B-Instruct", "revision_available": true, "status": "PASS"}}`

## Metrics

The JSON report contains complete per-provider and per-format records. Null or absent
metrics mean the corresponding real provider did not run; no value is inferred.

Confidence distribution: `{"count": 0, "max": null, "mean": null, "min": null, "values": []}`

## Evidence boundaries

- Synthetic, de-identified fixtures are used for decoder and post-processing checks.
- Automated tests may fake only the external checkpoint boundary and are not live-model proof.
- No model is downloaded or selected by this runner.
- OCR error rates require an approved ground-truth corpus; none is inferred from fixture text.

## Screenshot evidence

- dashboard: PASS - The existing developer page rendered on the screenshot-only server.
- swagger: PASS - Swagger rendered and exposed GET /api/v1/ocr/health.
- health: NOT VERIFIED - The actual health operation returned 503 because providers were not initialized.
- ocr_output: NOT VERIFIED - A synthetic upload produced no OCR output because the runtime was not ready.

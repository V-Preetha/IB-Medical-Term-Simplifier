# Phase 2 Architecture Convergence Report

Date: 2026-08-04  
Scope: OCR architecture correction only  
Result: **PASS for architecture convergence; Phase 2 remains IN PROGRESS**

## Approved executable path

```text
POST /api/v1/ocr
  -> OCRApplicationService
  -> BaseOCRProvider (registry/factory-selected Qwen3-VL; OCR + document type)
  -> BasePostProcessor (regex -> medical dictionary -> SymSpell)
  -> OCRResultBuilder
  -> OCRResponse
```

FastAPI routes resolve `OCRApplicationService` through dependency injection. The service
receives provider interfaces, a unit-of-work interface, a cache interface, and the result
builder. Production composition creates providers only through `ProviderRegistry` and the
OCR and post-processing factories. No API route imports or invokes a model library.

## Removed components

- Removed the complete `app.ingestion` implementation, including `ReportIngestionPipeline`,
  PaddleOCR, TrOCR, routing, OCR services, classifiers, PDF extraction, manual-review,
  cache, and ingestion models.
- Removed the parallel `app.normalization` pipeline and its rule files. Medical OCR
  post-processing now exists only behind `BasePostProcessor` in `app.ocr`.
- Removed legacy `app.api`, `app.jobs`, and `app.schemas` modules and the obsolete static
  single-page interface.
- Removed legacy ingestion, routing, normalization, manual-review, async-job, and API tests.
- Removed the legacy pipeline benchmark.
- Removed the obsolete provider skeleton compatibility module and stale provider-health
  router; production implementations and the converged router are imported directly.
- Removed `paddleocr`, `paddlepaddle`, and `numpy` from runtime dependencies. Removed the
  obsolete TrOCR optional dependency group.

## Refactored and connected components

- Added `OCRApplicationService` as the sole business orchestration boundary.
- Connected configuration-driven provider registry, discovery, factories, lifecycle, and
  FastAPI dependency injection to application startup.
- Added bounded process-local repository/cache adapters implementing Phase 2 interfaces;
  PostgreSQL, Redis, and Celery were not introduced.
- Added a versioned cache identity containing tenant, content digest, provider/model/prompt/
  rule identity, pipeline version, and schema version. Content hashes are not returned by
  the public API.
- Added the required create/get/status/delete/health/models endpoints plus privacy-safe
  recent-request and log inspection endpoints used by the engineering console.
- Added complete OCR response provenance, confidence, timing, page count, warning, review,
  cache, model, rule, schema, and trace metadata.
- Added a same-origin Jinja2, Bootstrap 5, and vanilla-JavaScript engineering console.
- Added JSON structured logging and a bounded privacy-safe engineering log adapter.

## Runtime dependency audit

| Dependency | Retention reason |
| --- | --- |
| FastAPI | Versioned REST API, OpenAPI, dependency injection, and lifecycle |
| Jinja2 | Server-rendered engineering console |
| Pillow | Image validation, decoding, resizing, and multi-frame TIFF processing |
| pillow-heif | Approved HEIC/HEIF decoding through Pillow |
| PyMuPDF | Validated, ordered, multi-page PDF rendering |
| python-multipart | Streaming multipart upload parsing in FastAPI |
| PyTorch | Qwen3-VL tensor/device inference runtime |
| Transformers | Approved Qwen3-VL model/processor adapter |
| Uvicorn | Production ASGI process runtime |

The optional `clinical-ner` group remains because GLiNER is an approved future Phase 5
dependency already isolated from the OCR runtime. It is not installed or invoked by Phase 2.
Development-only HTTPX, pytest, Ruff, and psutil support API testing, static checks, and
the existing OCR validation/benchmark tooling.

## Verification evidence

- Ruff: PASS (`python -m ruff check app tests benchmarks`).
- Complete automated suite: PASS, 42 passed and 1 CUDA-only test skipped on the CPU-only
  PyTorch runtime.
- Static compilation: PASS (`python -m compileall -q app tests benchmarks`).
- Import walk: PASS; every importable `app` module resolved.
- Legacy source scan: PASS; no PaddleOCR, TrOCR, `ReportIngestionPipeline`, `app.ingestion`,
  or `app.normalization` reference remains in application code, tests, benchmarks, or
  dependency manifests.
- OpenAPI: PASS; required operations have summaries, descriptions, response schemas, and
  documented response codes.
- API and console: PASS with lifecycle-owned synthetic providers at the model-library
  boundary. Upload, result, status, delete, provider health, models, service liveness,
  service readiness, recent requests, logs, dashboard, and Swagger are exercised.

## Remaining Phase 2 work

Architecture convergence is complete, but Phase 2 is not marked complete. Live loading and
inference with the approved immutable Qwen3-VL artifact remains NOT VERIFIED.
The current host lacks an activated approved model configuration, and its installed
PyTorch build is CPU-only. Representative accuracy/confusion evidence,
live CPU/CUDA performance evidence, and the roadmap's reproducible container acceptance
criterion also remain open. These limitations are preserved in the validation reports and
must be resolved before Step 4 or Phase 2 can be completed.

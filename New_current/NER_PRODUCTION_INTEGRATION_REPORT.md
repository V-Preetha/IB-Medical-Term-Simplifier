# Phase 5 Stage 3 Production Integration Report

Status: **INTEGRATION PASS - CLINICAL ACCEPTANCE PENDING**  
Validated: 2026-08-04

## Approved model

| Field | Evidence |
| --- | --- |
| Provider | `biomedical-ner-all` |
| Repository | `d4data/biomedical-ner-all` |
| Immutable revision | `015a4050c9ac99722e61c547aa9b4282bcbedc7f` |
| License | Apache-2.0 |
| Local checkpoint | `New_current/.model-cache/ner/biomedical-ner-all` |
| Observed artifact-tree SHA-256 | `6884545271a453eed4ff94f5e718105154c83104a7188bdcf424693ab6768493` |
| Framework | Transformers token classification |
| Validated device | CPU |

The observed tree hash is local validation evidence, not an authoritative expected hash.
No model was downloaded, resolved from a moving revision, or substituted during production
integration or verification.

## Production architecture

```text
POST /api/v1/ner
  -> MedicalNERService
  -> BaseNERProvider
  -> production-only NERProviderRegistry
  -> BiomedicalNERProvider
  -> local AutoTokenizer and AutoModelForTokenClassification
  -> canonical entity schema
```

The production registry contains one provider. OpenMed GLiNER and ModernBERT are archived
in the model manifest; their providers, scoring service, resource monitor, and runner live
beneath `benchmarks/ner/`. They are excluded from application imports, FastAPI routes,
startup composition, dependency injection, lifecycle, health, and
`GET /api/v1/ner/models`.

## API and dashboard verification

| Gate | Result | Evidence |
| --- | --- | --- |
| `POST /api/v1/ner` | PASS | HTTP 200 with entities, offsets, token probabilities, aggregate confidence, latency, provider/model/revision, device, token throughput, warnings, and version metadata |
| `GET /api/v1/ner/health` | PASS | HTTP 200; overall `HEALTHY`; provider `ready` |
| `GET /api/v1/ner/models` | PASS | HTTP 200; exactly one provider, `biomedical-ner-all` |
| Benchmark route excluded | PASS | `/api/v1/ner/benchmark` absent from OpenAPI |
| Swagger | PASS | `/docs` returned HTTP 200 |
| OpenAPI | PASS | `/openapi.json` returned HTTP 200 and documents production POST/health/models operations |
| Engineering dashboard | PASS | `/ner` returned HTTP 200 with OCR text input, entity visualization/table, health, metadata, JSON response, and Swagger link |
| Structured logging | PASS | Registration, initialization, inference, and shutdown events contain request/model/revision/timing/confidence/device fields and no clinical text |

## Live model evidence

The pinned checkpoint initialized locally on CPU in `4305.866 ms`. A production API call
over the synthetic text `The patient has type 2 diabetes and takes metformin.` completed in
`49.021 ms`, returned HTTP 200, and reported aggregate entity confidence `0.872816` using
the mean entity softmax probability method. The response identified the exact immutable
revision and recorded 14 processed tokens at `289.663 tokens/s`.

The checkpoint returned `2 diabetes` as Disease and `met` as Medication for that sentence.
These offsets accurately reference the returned spans, but the partial spans demonstrate
that successful runtime integration is not equivalent to clinical quality acceptance.
The response also explicitly reported the ignored unmapped checkpoint label `B-History`.

## Verification gates

- Ruff: **PASS**.
- Full automated suite: **58 passed**, **1 CUDA-conditional skipped**, no failures.
- Python application and benchmark compilation/import verification: **PASS**.
- Production model local initialization: **PASS**.
- Production CPU inference: **PASS**.
- Overlapping-window execution: **PASS**; a 669-token synthetic input crossed the
  configured 512-token window and all returned entity spans matched source offsets.
- CUDA inference: **NOT VERIFIED** because installed PyTorch is CPU-only.
- Database migration: not applicable; no database code or schema changed.
- Redis/Celery: not applicable to Stage 3 and remain Phase 4 work.

## Configuration and safe failure

Production configuration is environment-backed through `NER_CONFIG__PROVIDER`,
`NER_CONFIG__MODEL_NAME`, `NER_CONFIG__MODEL_REVISION`, `NER_CONFIG__CACHE_DIR`,
`NER_CONFIG__DEVICE`, `NER_CONFIG__CONFIDENCE_THRESHOLD`, `NER_CONFIG__MAX_TOKENS`,
`NER_CONFIG__STRIDE_TOKENS`, and `NER_CONFIG__LABEL_MAPPING_JSON`. Provider and model
identity must match the approved
manifest entry. The tokenizer and model use `local_files_only=True`; missing or mismatched
artifacts fail closed.

The response confidence is uncalibrated and is identified by calibration version
`uncalibrated-biomedical-ner-all-v1`. Empty or below-threshold extraction is marked for
review rather than assigned fabricated confidence.

## Known limitations and disposition

- The accepted Stage 2 benchmark used four synthetic records and does not constitute a
  representative clinical-corpus acceptance test.
- Confidence calibration and clinical per-entity acceptance thresholds remain open.
- Long inputs use overlapping tokenizer windows with source-relative offsets and
  deterministic overlap ownership. Representative clinical chunk-boundary quality remains
  an acceptance gate.
- CUDA remains unverified. Phase 3 persistence and Phase 4 cache/worker integration have
  not been started or implemented by this task.
- The model's partial-span behavior requires clinical-corpus review before its entities are
  used without a review policy downstream.

## Decision

The Phase 5 Stage 3 synchronous production service integration is **PASS**. Phase 5 remains
`IN PROGRESS` until representative clinical quality thresholds, confidence calibration,
long-input behavior, and the separately planned persistence/cache/worker integrations are
completed. No Entity Linking, Relation Extraction, or Phase 6 work was introduced.

# Production Medical NER

Phase 5 Stage 3 exposes the approved `d4data/biomedical-ner-all` checkpoint through a
provider-neutral production boundary:

```text
POST /api/v1/ner -> MedicalNERService -> BaseNERProvider -> BiomedicalNERProvider
```

The production registry contains exactly one provider. Its repository identity, immutable
revision, Apache-2.0 license, and local checkpoint path are governed by
`MODEL_MANIFEST.md`. `NER_CONFIG__PROVIDER` may only name the manifest's approved winner;
a mismatch fails closed. Model and tokenizer loading uses the configured local directory
with `local_files_only=True` and never downloads or substitutes a checkpoint.

Production configuration variables are:

- `NER_CONFIG__PROVIDER`
- `NER_CONFIG__MODEL_NAME`
- `NER_CONFIG__MODEL_REVISION`
- `NER_CONFIG__CACHE_DIR`
- `NER_CONFIG__DEVICE`
- `NER_CONFIG__CONFIDENCE_THRESHOLD`
- `NER_CONFIG__MAX_TOKENS`
- `NER_CONFIG__STRIDE_TOKENS`
- `NER_CONFIG__LABEL_MAPPING_JSON`

Outputs use Disease, Symptom, Medication, Procedure, Anatomy, Laboratory Test,
Measurement, and Medical Abbreviation. Confidence is measured from model softmax
probabilities; the aggregate response confidence is the mean confidence of returned
entities and remains explicitly uncalibrated. Empty or low-confidence output requires
review rather than receiving a fabricated score.

Long inputs are processed as overlapping tokenizer windows. Window offsets remain relative
to the complete input, and overlap ownership plus exact-span deduplication prevents duplicate
entities from adjacent windows.

`GET /api/v1/ner/health` reports production readiness and `GET /api/v1/ner/models` returns
only the approved model. The same-origin engineering dashboard is available at `/ner`.
Clinical text is not written to normal structured logs.

The Stage 1/2 candidate runner and its reports remain under `benchmarks/ner/`. It is not
included in FastAPI routes, application startup, dependency injection, lifecycle, health,
or the production model API.

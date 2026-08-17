# IB Health Medical Term Simplifier Architecture

Status: living architecture document  
Version: 1.5  
Last updated: 2026-08-05

## 1. Purpose

This document is the single source of truth for the Medical Term Simplifier's target
architecture and the boundary between implemented and planned capabilities. It governs
production development in the IB Health ecosystem.

Repository governance is split deliberately: `AGENTS.md` supplies standing engineering
rules, this file owns architecture decisions, `ROADMAP.md` owns implementation order and
acceptance gates, and `IMPLEMENTATION_LOG.md` records verified delivery history. A roadmap
status or log entry must not override an architectural decision in this file.

The product accepts laboratory reports, discharge summaries, prescriptions, radiology
reports, and consultation notes. It produces medically faithful, patient-friendly
explanations, with optional translation and speech output.

The system assists comprehension. It does not diagnose, prescribe, replace a clinician,
or guarantee that generated content is clinically complete. All stages must preserve
provenance and fail safely when confidence or verification is inadequate.

## 2. Architecture status

The repository contains multiple generations of work. `New_current/` is the current
production-oriented service; the remaining top-level training, evaluation, and pipeline
directories are supporting or experimental artifacts unless a task says otherwise.

| Capability | Current repository state | Target state |
|---|---|---|
| API | FastAPI endpoints in `New_current/app/api.py`; synchronous and process-local asynchronous flows | Versioned `/api/v1/` module APIs behind a stable gateway |
| Document classification | Merged into Qwen3-VL's prompted generation; the standalone classifier provider was removed on 2026-08-04 | Qwen3-VL OCR provider returns document type with OCR output (achieved) |
| Digital PDF extraction | PyMuPDF | Retained as a fast path before OCR |
| Printed OCR | Qwen3-VL provider (PaddleOCR was removed from the production path on 2026-08-03; see `IMPLEMENTATION_LOG.md`) | Qwen3-VL provider (achieved) |
| Handwriting OCR | Handled by the same Qwen3-VL provider as printed text (TrOCR was removed on 2026-08-03); a dedicated handwriting-confidence manual-review workflow is not confirmed implemented | Replaceable provider; retain review workflow |
| OCR post-processing | Deterministic normalization and medical rules | SymSpell, medical abbreviation dictionary, and regex normalization behind a provider/service boundary |
| Medical NER | Production service in `New_current/app/ner/` | Approved pinned `d4data/biomedical-ner-all` provider |
| Simplification | Pinned local Qwen3-0.6B three-level provider/service, source-grounding guard, versioned API, and engineering console | Clinical benchmark acceptance and Phase 3/4 persistence/cache activation |
| Durable data | Async SQLAlchemy models/repositories, Alembic migration, health, jobs API, and Docker topology in `New_current/`; live PostgreSQL validation pending | PostgreSQL, SQLAlchemy, and Alembic using section 7 |
| Cache/jobs | Encrypted Redis adapter, versioned keys/locks, durable Celery tasks/recovery, health, and dashboard implemented; live Redis/worker validation pending | Redis-backed cache/state and Celery workers |
| Medical embeddings | Provider, registry, service, versioned API, health, and console in `New_current/app/embeddings/`; exact runtime artifact pending | BioClinical ModernBERT local embeddings with exact identity governed by `MODEL_MANIFEST.md` |
| Vector retrieval | Not implemented | Qdrant with a separately approved storage/retrieval integration |
| Entity linking | Provider, service, versioned API, health, and engineering console in `New_current/app/entity_linking/`; real runtime not configured | SciSpaCy + locally provisioned licensed UMLS artifacts with exact versions governed by `MODEL_MANIFEST.md` |
| Relation extraction | Provider, registry, service, versioned API, and fail-closed BioLinkBERT adapter in `New_current/app/relation_extraction/`; cached base encoder has no relation head | Fine-tuned BioLinkBERT relation-classification checkpoint with named ontology and calibrated thresholds |
| Verification | Local PubMedBERT/MedNLI technical boundary, deterministic grounding checks, versioned API, and testing page implemented; license and production approval pending | PubMedBERT/MedNLI |
| Translation/speech | IndicTrans2 provider/service/versioned API and pinned local CUDA runtime model-validated; representative clinical-quality validation remains open. Kokoro TTS is Deferred for MVP. | IndicTrans2 and Kokoro TTS |
| Developer UI | Same-origin stage consoles plus consolidated `/engineering-demo` Jinja2/Bootstrap/vanilla-JavaScript dashboard | Internal engineering surfaces remain separate from the customer frontend |

Planned components must not be described as deployed. A model becomes "selected" only
after benchmark evidence, an architecture decision, configuration, tests, and deployment
metadata are committed.

## MVP pipeline overlay

The MVP does not replace the target architecture below. It temporarily orchestrates only:

```text
Upload -> OCR -> Medical NER -> Qwen3 Simplification -> IndicTrans2 Translation
       -> final patient-friendly report
```

Medical embeddings are background-only and non-blocking. Entity Linking, Relation
Extraction, Medical Verification, and TTS are **Deferred for MVP**; their existing
boundaries remain registered and documented but are not invoked by the MVP demo flow.

## 3. Target pipeline

```text
Upload
  -> Qwen3-VL OCR and document-type inference
  -> regex normalization
  -> medical abbreviation dictionary
  -> SymSpell
  -> medical entity recognition
  -> entity linking (optional while the licensed runtime is pending)
  -> relation extraction
  -> medical embeddings
  -> vector retrieval
  -> LLM simplification
  -> medical verification
  -> translation (optional)
  -> text-to-speech (optional)
  -> patient output
```

The orchestrator passes typed stage results, provenance, model version, confidence, and
timing. A stage must not reach into another stage's model implementation. Optional stages
must be explicit in the response rather than silently skipped.

## 4. Target model and platform stack

| Stage | Target technology | Selection note |
|---|---|---|
| OCR and document-type inference | Qwen3-VL-4B-Instruct | Sole Phase 2 model; exact immutable revision is governed by `MODEL_MANIFEST.md` |
| OCR post-processing | SymSpell + medical abbreviation dictionary + regex normalization | Protect measurements, medication names, negation, and uncertain text |
| Medical NER | `d4data/biomedical-ner-all` | Approved production winner at revision `015a4050c9ac99722e61c547aa9b4282bcbedc7f`; consumers remain provider-neutral |
| Entity linking | SciSpaCy + UMLS Linker | UMLS licensing and deployment access are prerequisites |
| Relation extraction | `michiyasunaga/BioLinkBERT-base` backbone | Architecture approved; runtime requires a fine-tuned relation-classification head and must not use randomly initialized weights |
| Medical embeddings | BioClinical ModernBERT | Provider boundary implemented; repository ID, immutable revision, license, and local artifact remain pending approval |
| Vector database | Qdrant | Collections must be versioned by embedding model and schema |
| Simplification | `Qwen/Qwen3-0.6B` | Approved local-only revision `c1899de289a04d12100db370d81485cdf75e47ca`; prompt identity is governed by `MODEL_MANIFEST.md` |
| Medical verification | PubMedBERT fine-tuned on MedNLI | Verification flags or rejects; it does not silently rewrite facts |
| Translation | IndicTrans2 | Preserve entities, values, units, warnings, and provenance |
| Speech | Kokoro TTS | Planned; language/voice support must be validated |
| Relational database | PostgreSQL | Existing logical schema in section 7 |
| Cache and coordination | Redis | Do not store large report bodies when encrypted object storage is appropriate |
| Background tasks | Celery | Dedicated GPU queues; idempotent tasks |

Model names identify architecture intent, not permission to download or send clinical
data to third parties. Production providers must use approved hosting and record exact
model, tokenizer, prompt, and pipeline versions.

## 5. Service boundaries

Every stage follows the same dependency direction:

```text
HTTP route -> application service -> provider interface -> model/infrastructure adapter
                     |                       |
                     +-> domain schemas      +-> environment-backed configuration
```

Each module owns:

- versioned routes and OpenAPI documentation;
- request, response, and internal result schemas;
- an application service that applies policy and orchestration;
- one provider interface and one or more replaceable adapters;
- configuration, typed errors, health/readiness behavior, logs, metrics, and tests.

Routes never import model SDKs. Domain/application code never imports FastAPI, Redis,
Celery, Qdrant, or a model library directly. Infrastructure adapters implement inward-
facing interfaces and are wired at application startup.

Large uploads and results belong in approved encrypted object storage; PostgreSQL stores
metadata and references. Redis coordinates cache keys, idempotency, progress, and jobs.
Celery workers run expensive stages, with one inference worker per GPU unless capacity
testing proves another topology safe.

## 6. API conventions

New public APIs use `/api/v1/`. Representative resource boundaries are:

```text
POST   /api/v1/reports
GET    /api/v1/reports/{report_id}
DELETE /api/v1/reports/{report_id}
GET    /api/v1/reports/{report_id}/status
GET    /api/v1/reports/{report_id}/result

POST   /api/v1/ocr
POST   /api/v1/ner
POST   /api/v1/entity-linking
GET    /api/v1/entity-linking/health
GET    /api/v1/entity-linking/models
POST   /api/v1/relation-extraction
GET    /api/v1/relation-extraction/health
GET    /api/v1/relation-extraction/models
POST   /api/v1/embeddings
GET    /api/v1/embeddings/health
GET    /api/v1/embeddings/models
POST   /api/v1/simplify
GET    /api/v1/simplify/health
GET    /api/v1/simplify/models
POST   /api/v1/jobs
GET    /api/v1/jobs
GET    /api/v1/jobs/{job_id}
DELETE /api/v1/jobs/{job_id}
GET    /api/v1/infrastructure/health
POST   /api/v1/simplifications  (deprecated compatibility operation)
POST   /api/v1/translations
POST   /api/v1/speech

GET    /api/v1/health/live
GET    /api/v1/health/ready
```

Only expose GET, POST, PUT/PATCH, or DELETE when the resource semantics require it; do
not add meaningless methods merely for symmetry. Existing unversioned endpoints in
`New_current/` are compatibility surfaces until a separately approved migration.

All endpoints require a concise summary and description, typed Pydantic request and
response models, realistic de-identified examples, documented success/error codes, and
a stable error envelope containing `code`, safe `message`, and `request_id`. Long work
returns `202 Accepted` and a pollable job resource. Duplicate submissions are idempotent
by tenant, content hash, and pipeline version.

Propagate a validated request ID. Authentication and authorization are enforced at the
platform boundary and again for resource ownership. Never reveal existence of another
tenant's report.

## 7. Existing database schema

The schema below transcribes `DB.pdf`. It is authoritative at the logical level and must
be reused. Physical constraints, indexes, enums, retention columns, and audit mechanics
may be added through Alembic when required, without changing the model's meaning.

### Core report processing

| Table | Columns |
|---|---|
| `users` | `user_id uuid` PK; `full_name varchar`; `email varchar`; `created_at timestamp` |
| `reports` | `report_id uuid` PK; `user_id uuid` FK; `original_filename varchar`; `storage_url text`; `file_type varchar`; `language varchar`; `upload_time timestamp`; `status varchar`; `page_count int`; `file_size bigint` |
| `report_processing` | `process_id uuid` PK; `report_id uuid` FK; `pipeline_version varchar`; `started_at timestamp`; `completed_at timestamp`; `processing_status varchar`; `processing_time_seconds float`; `ocr_required boolean`; `ocr_success boolean`; `raw_extracted_text text` |
| `medical_entities` | `entity_id uuid` PK; `process_id uuid` FK; `entity_text text`; `entity_type varchar`; `start_offset int`; `end_offset int`; `confidence float` |
| `simplifications` | `simplification_id uuid` PK; `process_id uuid` FK; `predicted_level varchar`; `prediction_confidence float`; `simplified_text text`; `executive_summary text`; `created_at timestamp` |
| `model_outputs` | `output_id uuid` PK; `process_id uuid` FK; `model_name varchar`; `model_version varchar`; `stage varchar`; `execution_time float`; `token_usage int`; `status varchar` |
| `feedback` | `feedback_id uuid` PK; `simplification_id uuid` FK; `rating int`; `comments text`; `created_at timestamp` |

### Preferences and speech

| Table | Columns |
|---|---|
| `voice_profiles` | `voice_profile_id uuid` PK; `user_id uuid` FK; `preferred_language varchar`; `preferred_accent varchar`; `preferred_dialect varchar`; `voice_sample_url text`; `accent_detected boolean`; `created_at timestamp`; `updated_at timestamp` |
| `voice_generations` | `generation_id uuid` PK; `simplification_id uuid` FK; `voice_profile_id uuid` FK; `tts_model varchar`; `audio_url text`; `duration_seconds float`; `generation_status varchar`; `created_at timestamp` |
| `supported_dialects` | `dialect_id uuid` PK; `language varchar`; `accent varchar`; `dialect_name varchar`; `region varchar`; `is_active boolean` |
| `user_preferences` | `preference_id uuid` PK; `user_id uuid` FK; `health_literacy_level varchar`; `preferred_language varchar`; `accessibility_mode boolean`; `created_at timestamp` |

### Relationships transcribed from the diagram

- A user owns many reports; each report belongs to one user.
- A report may have many processing attempts; each attempt belongs to one report.
- A processing attempt may produce many medical entities and model outputs.
- A processing attempt may produce one simplification; a simplification belongs to one
  processing attempt.
- A simplification may receive many feedback entries and may have many voice generations.
- A user may have one voice profile and one preferences record.
- A voice profile may be used by many voice generations.
- `supported_dialects` is reference data and has no explicit foreign-key relationship in
  the source diagram.

All schema changes require SQLAlchemy model updates and an Alembic migration. Do not put
binary uploads or generated audio in PostgreSQL. Store approved encrypted object URLs and
apply deletion/retention policies to both metadata and objects.

### Phase 3 physical extensions

The production ORM retains the logical names above. Requested document/OCR/NER concepts
map to `reports`, `report_processing`/`model_outputs`, and `medical_entities`; duplicate
tables with competing meanings are prohibited. The initial physical migration adds only
the resources not represented by the diagram: `entity_links`, `embedding_records`
(metadata only), `translations`, `processing_jobs`, append-only `audit_logs`, and
`model_registry`. It also adds SHA-256 document identity and common UUID/timestamp/version/
soft-delete mechanics needed for idempotency, traceability, and retention.

## 8. Cache, retrieval, and version identity

Use SHA-256 for uploaded-file cache identity. Cache keys must also include tenant scope,
stage, normalized configuration, pipeline version, exact model/tokenizer version, and
prompt/rule version where relevant. Never expose hashes as public identifiers.

Redis is used for OCR, NER, embedding, translation, prompt, and session caches, as well
as job state and single-flight coordination. Store only the minimum sensitive payload,
encrypt transport and storage, enforce TTLs, and invalidate on version changes.

The implemented Redis adapter encrypts stage JSON before storage and includes tenant,
document SHA-256, stage, pipeline, immutable model revision, configuration, prompt/rule,
and schema versions in cache identity. Celery uses late acknowledgement, worker-loss
rejection, bounded retries with exponential backoff, CPU/GPU queues, durable PostgreSQL
job state, and a Beat recovery task for broker submission failures. Stage executors are
deployment entry points, preserving inward dependency direction and the frozen AI modules.

Qdrant collections are isolated by environment and tenant policy. Each point records the
embedding model/version, source report/process, chunk provenance, and access-control
metadata. Retrieval must filter authorization before results reach the LLM.

## 9. Reliability, observability, and performance

Log structured events with request ID, tenant-safe resource ID, stage, latency,
processing time, cache hit/miss, model name/version, confidence, and safe CPU/GPU/memory
metrics. Raw report text, prompts, simplified text, tokens, credentials, and signed URLs
must not appear in normal logs.

Measure queue delay separately from execution time. Record stage timings and model output
metadata using `report_processing` and `model_outputs` where applicable. Liveness checks
only process health; readiness checks required dependencies, model availability, and
worker capacity without executing expensive inference on every probe.

Expensive deterministic work should be cached. Load models once per worker, batch when
latency objectives allow, bound queues and uploads, use timeouts/cancellation, and make
Celery tasks retry-safe and idempotent. Retry transient infrastructure failures only;
invalid input and clinical verification failures are terminal unless corrected.

## 10. Security and clinical safety

- Detect type from content, enforce allowlists and size/page limits, and reject malformed,
  encrypted, decompression-bomb, or unsupported documents.
- Use isolated temporary storage, safe generated names, least-privilege access, and
  guaranteed cleanup. Scan uploads according to the platform security policy.
- Encrypt data in transit and at rest; keep secrets in the approved secret manager.
- Enforce tenant authorization on database records, object storage, cache entries, jobs,
  and vector retrieval.
- Maintain auditability and configured retention/deletion without logging clinical text.
- Preserve negation, uncertainty, numeric values, reference ranges, units, medication
  names/doses, chronology, and links from simplified claims to source evidence.
- Treat low-confidence OCR, unresolved entities, retrieval misses, verification
  contradictions, and unsupported translation/speech as explicit states. Route to manual
  review or return a safe partial result; do not fabricate missing content.
- Use synthetic or approved de-identified test data only.

## 11. Testing and delivery standards

Each module includes unit tests for domain/service behavior, provider contract tests,
integration tests for infrastructure adapters, API/OpenAPI tests, failure and timeout
tests, cache/idempotency tests, and health/readiness tests. AI stages additionally need
versioned benchmark datasets and thresholds for accuracy, faithfulness, latency, and
resource usage.

Production changes require:

1. typed, formatted, lint-clean code and focused docstrings;
2. environment-backed configuration with safe defaults;
3. deterministic tests that do not require downloading models unless explicitly marked;
4. migrations for schema changes and a rollback strategy;
5. updated API examples and this document when a decision changes;
6. verification results and explicit disclosure of anything not run.

## 12. Developer interface

The target internal console is served by FastAPI using Jinja2 templates, Bootstrap 5,
and vanilla JavaScript; it has no separate frontend server. Per module, it should let
engineers submit de-identified inputs, inspect structured output, confidence, provenance,
latency, safe logs/trace IDs, raw API responses, and downloadable artifacts, and open the
Swagger UI. It is not a patient-facing interface and must not bypass normal authentication,
authorization, upload validation, or retention controls.

The consolidated `/engineering-demo` page is an internal demonstration surface. It
orchestrates only existing REST endpoints from the browser, labels unavailable stages
explicitly, and contains no independent model, database, cache, job, or business logic.
Planned and frozen roadmap stages are static delivery-state cards until their approved
APIs exist; they must not be represented by fabricated endpoint responses.

## 13. Medical report simplification boundary

Phase 9 follows the standard dependency direction: the HTTP route depends on
`SimplificationService`, which depends only on `BaseSimplificationProvider`; only the
Qwen3 adapter imports model libraries. The provider resolves repository ID, immutable
revision, and prompt version from `MODEL_MANIFEST.md` with exact-match environment
overrides, uses local files only, loads the checkpoint once, and generates all three
readability levels in one deterministic inference.

The versioned prompt is an external JSON resource and source text is delimited as
untrusted data. The service rejects output that adds numeric facts/units or explains terms
absent from source evidence. Confidence is a source-fact/entity preservation ratio, not a
calibrated clinical probability; every result requires review. The API does not silently
fall back, invoke a remote model, or fabricate a successful response.

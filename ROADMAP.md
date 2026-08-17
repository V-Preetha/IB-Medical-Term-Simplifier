# Medical Term Simplifier Roadmap

Status: engineering contract  
Last updated: 2026-08-05

## Contract

This file defines implementation order and completion. `ARCHITECTURE.md` defines what the
system is; this roadmap defines when it is built. Do not begin a later phase until the
preceding phase is complete unless the task explicitly changes the roadmap.

Status values:

- `COMPLETE`: every deliverable and acceptance criterion is verified.
- `IN PROGRESS`: the active phase; some criteria remain open.
- `PLANNED`: not started under this roadmap contract.
- `BLOCKED`: cannot proceed without a recorded external decision or dependency.

Pre-existing or experimental code is useful baseline work but does not complete a phase
until it meets the phase's full contract. On completion, update the status and add an
entry to `IMPLEMENTATION_LOG.md` with verification evidence and remaining limitations.
## MVP delivery override

The active MVP sequence is Upload Report -> OCR -> Medical NER -> Qwen3 Simplification
-> IndicTrans2 Translation -> final patient-friendly report. Medical embeddings may run
in the background and must not block the patient workflow. Entity Linking, Relation
Extraction, Medical Verification, and TTS are **Deferred for MVP**. Their production
architecture, APIs, provider boundaries, and roadmap acceptance criteria are preserved;
no deferred runtime is called by the MVP orchestrated demo.


## Phase 1 - Repository Infrastructure

Status: `COMPLETE`

### Deliverables

- Standing engineering context in `AGENTS.md`.
- Architecture source of truth in `ARCHITECTURE.md`.
- Ordered engineering contract in `ROADMAP.md`.
- Persistent history in `IMPLEMENTATION_LOG.md`.
- Production source boundary identified as `New_current/`.
- Python project metadata, environment-backed configuration, test layout, and lint rules.

### Acceptance criteria

- A new contributor can identify the production source, architecture, active phase, and
  completion rules without a large task prompt.
- Current and target implementations are clearly distinguished.
- Existing database schema and non-negotiable healthcare safety rules are documented.
- Future phases cannot be marked complete without verification evidence.

## Phase 2 - OCR Service

Status: `IN PROGRESS`

### Incremental implementation progress

- [x] Step 1: establish the isolated OCR service and test package structure under
  `New_current/` without changing existing compatibility behavior.
- [x] Step 2: define persistence-neutral, tenant-scoped repository and unit-of-work
  interfaces; concrete database integration remains in Phase 3.
- [x] Step 3: add configuration-driven provider contracts, registry/discovery, factories,
  lifecycle management, FastAPI dependency injection, and provider health metadata;
  model inference remains unimplemented for the later integration steps.
- [x] Architecture convergence: remove the competing legacy ingestion path and connect one
  executable OCRApplicationService pipeline, versioned REST API, structured logging, and
  same-origin Jinja2 engineering console.
- [ ] Step 4: production Qwen3-VL and regex/dictionary/SymSpell adapters are
  implemented and boundary-tested; completion awaits verification with approved immutable
  model revisions, representative medical documents, and deployment hardware.
- [ ] Production validation: `NOT VERIFIED` on 2026-08-03. The deterministic model
  manifest pins the approved Qwen3-VL repository and immutable revision. On 2026-08-04,
  the local checkpoint initialized on CPU and the simplified provider health check passed;
  complete live CPU format inference and clinical benchmark evidence remain open.
  CUDA is optional and `NOT VERIFIED` because the installed PyTorch build is CPU-only.
- [ ] Remaining Phase 2 deliverables and acceptance criteria below are not yet complete.

The approved model manifest values, immutable checkpoint identity, local availability,
and CPU initialization health are verified. The following list is the exclusive set of
remaining Phase 2 validation gates:

- complete CPU inference validation across supported Phase 2 formats;
- optional CUDA validation when a compatible runtime is available;
- OCR accuracy and performance benchmark; and
- clinical validation using an approved de-identified corpus and acceptance thresholds,
  including production post-processing resource approval.

Current implementation: `New_current/` exposes one OCR runtime path through
`OCRApplicationService`, registry/factory-selected Qwen3-VL, and medical post-processing
providers. Qwen3-VL returns OCR text and document type in one structured generation. Live
approved-model and deployment verification remains open.

### Deliverables

- Versioned `/api/v1/ocr` service and report-upload integration.
- Replaceable Qwen3-VL multimodal provider for OCR and prompted document-type inference.
- Explicit handwriting provider and low-confidence manual-review workflow.
- SymSpell, medical abbreviation dictionary, and regex post-processing with protected
  medical values, units, doses, negation, and uncertainty.
- SHA-256 cache interface with a functional local/test adapter ready for Redis in Phase 4.
- Typed schemas exposing trace IDs, report ID, model/revision, confidence method,
  processing time, cache status, pipeline version, warnings, and review state.
- Complete OpenAPI metadata, health/readiness endpoints, same-origin Jinja2/Bootstrap 5/
  vanilla-JavaScript test UI, unit/integration/API/failure tests, and container image.

### Acceptance criteria

- Accept validated PDF, PNG, JPEG, TIFF, BMP, WebP, and HEIC inputs and reject malformed,
  oversized, encrypted, or unsupported uploads safely.
- Correctly handle digital, scanned, handwritten, mixed, and multi-page documents.
- Preserve page order, numeric values, reference ranges, units, medication names/doses,
  negation, and source provenance.
- Exact duplicate content hits the cache and returns equivalent versioned output.
- Low-confidence output never silently proceeds; review/failure behavior is tested.
- Responses contain no mock text and include all AI inference contract metadata.
- Swagger examples and error codes are complete; liveness and readiness are tested.
- All phase tests and static checks pass.

## Phase 3 - Database Layer

Status: `IN PROGRESS - SOFTWARE IMPLEMENTED, LIVE POSTGRESQL VALIDATION PENDING`

### Implementation progress

- [x] Add async SQLAlchemy 2.x models and tenant-scoped repositories for the authoritative
  schema plus the minimum required Entity Linking, embedding metadata, translation, job,
  audit, and model-registry extensions.
- [x] Add UUID identities, timestamps, optimistic versions, appropriate soft deletion,
  ownership filters, constraints, indexes, object-reference-only document/audio storage,
  and transactional job behavior.
- [x] Add Alembic configuration and the reversible `0001_initial_schema` migration.
- [x] Verify metadata, repositories, tenant isolation, soft deletion, the single Alembic
  head, and PostgreSQL offline upgrade/downgrade SQL.
- [ ] Run upgrade, CRUD/rollback/outage integration tests, and downgrade against a real
  PostgreSQL service. PostgreSQL/Docker are unavailable on the current host.
- [ ] Bind the frozen OCR/NER stage results to the new repositories in a separately
  reviewed integration and verify atomic end-to-end persistence.

### Deliverables

- PostgreSQL integration using async SQLAlchemy repositories.
- Models matching the existing schema in `ARCHITECTURE.md` and `DB.pdf`.
- Alembic configuration and reviewed initial migration.
- Transactional CRUD services and versioned APIs for reports, processing attempts,
  entities, simplifications, feedback, preferences, voice profiles, and reference data
  where semantically applicable.
- Durable report/process IDs, ownership enforcement, object-storage references, indexes,
  constraints, retention hooks, health checks, and integration tests.

### Acceptance criteria

- Upgrade from an empty database and downgrade/rollback are verified.
- Foreign keys, uniqueness, validation, timestamps, and cascade/retention behavior match
  the documented schema and do not expose cross-tenant records.
- OCR creates traceable report, processing, model-output, and entity records atomically.
- CRUD behavior, conflicts, missing records, transaction rollback, and database outages
  are covered by tests and OpenAPI documentation.
- No binary report or audio payload is stored in PostgreSQL.

## Phase 4 - Redis and Celery

Status: `IN PROGRESS - SOFTWARE IMPLEMENTED, LIVE REDIS/CELERY VALIDATION PENDING`

### Implementation progress

- [x] Add encrypted Redis stage caching with tenant/version-complete keys, configurable
  TTL, metadata, invalidation, statistics, and token-safe single-flight locking.
- [x] Add durable Celery CPU/GPU queues and named OCR, NER, Entity Linking, Embeddings,
  Simplification, and Translation tasks behind deployment-owned stage-executor contracts.
- [x] Add retries with exponential backoff, time limits, late acknowledgement, worker-loss
  recovery, progress/failure/cancellation state, idempotent job submission, and a periodic
  recovery task for database-committed broker failures.
- [x] Add versioned job/health APIs, infrastructure dashboard, structured safe logging,
  Dockerfile, and Compose services for API, PostgreSQL, Redis, migration, worker, and Beat.
- [x] Pass protocol-compatible cache, queue configuration/task, API, dashboard, OpenAPI,
  repository, and job-service tests.
- [ ] Verify live Redis expiry/locking/outages, worker restart/retry/recovery, queue depth,
  and Docker Compose health convergence. Docker/Redis are unavailable on the current host.
- [ ] Register and review adapters from the frozen application services into the six Celery
  stage entry points, then verify real pipeline jobs. Missing adapters currently fail closed.

### Deliverables

- Redis adapters for caches, idempotency, job state, progress, and single-flight locks.
- Celery queues and workers for durable CPU/GPU processing.
- Versioned cache-key builder including tenant, content hash, stage, model, configuration,
  prompt/rule, schema, and pipeline versions.
- Retry, timeout, cancellation, TTL, dead-letter/failure, and graceful-shutdown policies.
- Docker Compose development topology for API, PostgreSQL, Redis, worker, and required
  health checks.

### Acceptance criteria

- Jobs survive API restarts and duplicate submissions execute inference once.
- Cache hit/miss, invalidation, expiry, stampede prevention, and tenant isolation work.
- Transient failures retry safely; invalid clinical inputs do not retry indefinitely.
- Worker progress and request IDs are traceable through API, Redis, Celery, and database.
- Redis/Celery outages produce safe, documented errors without losing acknowledged work.

## Phase 5 - Medical Entity Recognition

Status: `IN PROGRESS`

Stage 1 evaluation was explicitly authorized to proceed without renumbering phases or
starting production NER integration. Phase 3 remains the Database Layer and Phase 4
remains Redis and Celery.

### Stage 1 - Evaluation Framework

- [x] Provider-neutral entity/result/health contracts and an instance-scoped candidate
  registry for the three approved benchmark candidates.
- [x] Fail-closed `MODEL_MANIFEST.md` candidate inventory with no guessed repository IDs,
  moving revisions, model downloads, or production winner.
- [x] Evaluation application service normalizing eight medical entity types and measuring
  exact-span precision, recall, F1, entity-level accuracy, false positives, false negatives,
  inference latency, peak RAM, optional peak GPU memory, loading time, and tokens/second.
- [x] Evaluation-only `/api/v1/ner/benchmark`, `/api/v1/ner/models`, and
  `/api/v1/ner/health` endpoints with typed OpenAPI contracts and stable error handling.
- [x] Same-origin NER benchmark console, identical-input JSONL runner, synthetic template,
  and JSON/CSV/Markdown report artifacts.
- [x] Approve immutable identities, revisions, licenses, and local checkpoints for all
  three candidates.
- [ ] Run all three candidates on the same approved, representative, de-identified corpus
  and record per-entity and aggregate accuracy/performance results.
- [x] Review benchmark evidence and record a separate production model decision.

### Stage 2 - Candidate Evaluation

- [x] Execute all three immutable local checkpoints offline on the same synthetic,
  de-identified, eight-entity evaluation dataset and generate JSON, Markdown, overall CSV,
  per-entity CSV, and leaderboard CSV evidence.
- [x] Record measured exact-span quality, latency, loading time, peak RAM, conditional GPU
  memory, and tokens/second. CUDA remains `NOT VERIFIED` on the CPU-only runtime.
- [x] Recommend `biomedical-ner-all` for review based on the highest measured macro F1;
  the report records `winner: null` and no production provider has been integrated.
- [x] Obtain explicit approval of `d4data/biomedical-ner-all` as the production winner.
- [ ] Complete representative clinical-corpus, calibration, and threshold validation
  before final clinical acceptance.

### Stage 3 - Production Integration

- [x] Configure `d4data/biomedical-ner-all` revision
  `015a4050c9ac99722e61c547aa9b4282bcbedc7f` as the sole production provider while
  retaining the provider interface, registry, dependency injection, lifecycle, health,
  environment-backed configuration, and structured logging boundaries.
- [x] Remove candidate benchmarking from FastAPI startup and routes. The offline candidate
  runner and evidence remain available only beneath `New_current/benchmarks/ner/`.
- [x] Add production `POST /api/v1/ner`, `GET /api/v1/ner/health`, and
  `GET /api/v1/ner/models` operations with versioned schemas, exact offsets, measured
  confidence and latency, immutable model metadata, examples, and safe errors.
- [x] Replace the candidate console with the same-origin `/ner` production engineering
  dashboard for OCR-text selection/input, entity visualization, table output, model health,
  inference metadata, raw JSON, and Swagger access.
- [x] Verify the real pinned local model on CPU, production health, inference, OpenAPI,
  Swagger, dashboard, structured lifecycle logs, Ruff, tests, and import compilation.
- [ ] Complete representative clinical-corpus quality thresholds, confidence calibration,
  long-input/chunk-boundary validation, and Phase 3/4 persistence/cache/worker adapters.

Archived baseline GLiNER code is not composed into the production application. The only
Phase 5 runtime registry entry is the approved biomedical-ner-all provider.

### Deliverables

- Versioned `/api/v1/ner` service with replaceable provider interface.
- Reproducible benchmark of OpenMed Zero-Shot GLiNER, `biomedical-ner-all`, and
  `Kushtrim/ModernBERT-base-biomedical-ner` on representative de-identified data.
- Recorded model decision, exact revision, label ontology, confidence method,
  batching/chunking, offset provenance, database persistence, Redis caching, and tests.

### Acceptance criteria

- Selected model meets recorded per-entity precision/recall/F1 and latency thresholds.
- Entity spans map correctly to normalized and source text across chunk boundaries.
- Negated, uncertain, duplicate, overlapping, and low-confidence entities are explicit.
- API, worker, cache, persistence, confidence, model metadata, and failure paths pass tests.

## Phase 6 - Entity Linking

Status: `DEFERRED FOR MVP - ARCHITECTURE PRESERVED`

### Deliverables

- SciSpaCy + UMLS linker behind a replaceable provider.
- UMLS license/access procedure, terminology version pinning, candidate ranking,
  confidence, ambiguity handling, caching, persistence, APIs, and tests.

### Implementation progress

- [x] Add the provider interface, instance-scoped registry, dependency injection,
  application service, typed failures, structured lifecycle/inference logging, and
  local-only SciSpaCy UMLS production adapter.
- [x] Add versioned POST, health, and models APIs plus the same-origin engineering console
  and an OCR-text -> production NER -> entity-linking test flow without changing OCR or NER.
- [x] Preserve ranked candidates, original NER span provenance, CUI, preferred name,
  semantic type identifiers, SciSpaCy similarity confidence, UMLS source, ambiguity, and
  explicit unlinked states in the versioned response.
- [x] Add fail-closed manifest/environment configuration and tests for contracts, registry,
  service, API, health, dashboard, OpenAPI, invalid input, and missing production resources.
- [ ] Approve and record the exact SciSpaCy version, SciSpaCy language model and immutable
  version, licensed UMLS release, license procedure, and local artifact paths; provision
  those artifacts without implicit downloads.
- [ ] Run real UMLS inference, clinical accuracy/ambiguity benchmarks, confidence threshold
  approval, authorization tests, outage tests, and terminology-update validation.
- [ ] Add the roadmap-required durable persistence and shared cache adapters when the
  approved Phase 3 and Phase 4 infrastructure contracts are available.

### Acceptance criteria

- Links include concept identifier, preferred name, terminology/version, candidates,
  confidence method, and source entity provenance.
- Ambiguous and unlinked entities remain explicit and never receive fabricated concepts.
- Authorization, reproducibility, benchmark thresholds, cache behavior, and outages pass.

## Phase 7 - Relation Extraction

Status: `DEFERRED FOR MVP - ARCHITECTURE PRESERVED`

### Deliverables

- BioLinkBERT relation provider and versioned `/api/v1/relation-extraction` service.
- Relation ontology, entity-pair generation, evidence spans, confidence calibration,
  persistence, Redis caching, benchmark, and tests.

### Implementation disposition

- [x] Add the provider interface, instance-scoped registry, dependency injection,
  application service, lifecycle, environment/manifest configuration, structured logging,
  safe errors, versioned POST/health/models API contracts, and dynamic label ontology.
- [x] Pin the locally evidenced `michiyasunaga/BioLinkBERT-base` revision
  `b71f5d70f063d1c8f1124070ce86f1ee463ca1fe` and validate its checkpoint contract.
- [x] Reject the cached `BertModel` base encoder as `incompatible_artifact`; do not allow
  Transformers to initialize a random sequence-classification head or fabricate relations.
- [ ] Supply and approve a fine-tuned biomedical relation-extraction checkpoint with named
  `id2label`, explicit no-relation labels, the versioned entity-marker preprocessing
  contract, immutable revision evidence, and calibration metadata.
- [ ] Run real inference, clinical relation/negation/direction benchmarks, authorization,
  persistence, cache, and outage verification. Runtime remains intentionally deferred.

### Acceptance criteria

- Relations reference valid source entities and supporting text.
- Direction, negation, temporality, uncertainty, and cross-sentence behavior are tested.
- Unsupported or low-confidence relations are not presented as facts.
- Recorded quality and latency thresholds are met reproducibly.

## Phase 8 - Embeddings and Qdrant

Status: `IN PROGRESS - MVP BACKGROUND ONLY; MODEL RUNTIME PENDING`

### Deliverables

- BioClinical ModernBERT embedding provider.
- Qdrant collections, payload schema, tenant filters, indexing, versioning, ingestion,
  deletion, retrieval service, cache integration, APIs, and tests.

### Stage 1 - Medical Embeddings progress

- [x] Add the provider interface, registry, dependency injection, lifecycle, local-only
  Transformers adapter, structured logs, safe errors, health/model inventory, and
  versioned batch embedding responses.
- [x] Implement attention-mask mean pooling, optional L2 normalization, CPU/CUDA device
  policy, batch limits, vector dimensions/norms, token counts, latency/throughput, and
  exact reproducibility metadata.
- [x] Add `POST /api/v1/embeddings`, `GET /api/v1/embeddings/health`,
  `GET /api/v1/embeddings/models`, and the same-origin `/embeddings` engineering console.
- [x] Return vectors directly without Qdrant, persistence, cache, collection, retrieval,
  or indexing behavior.
- [ ] Approve the exact BioClinical ModernBERT repository ID, immutable revision, license,
  local cache, expected dimensions, and deployment thresholds; run real CPU/CUDA inference.
- [ ] Implement Qdrant only in the separately requested vector-storage stage. Phase 8
  remains `IN PROGRESS` until the existing Qdrant acceptance criteria are completed.

### Acceptance criteria

- Embeddings expose exact model/revision, dimensions, normalization, timing, and
  reproducibility metadata.
- Re-indexing and deletion are safe; incompatible model versions cannot share a collection.
- Authorization filters apply before retrieval results reach downstream models.
- Retrieval quality, latency, duplicate handling, and Qdrant outage behavior are verified.

## Phase 9 - Qwen Simplification

Status: `IN PROGRESS - PRODUCTION BOUNDARY IMPLEMENTED; CLINICAL VALIDATION OPEN`

Baseline: `New_current/` contains an embedded Qwen3-0.6B simplifier. Preserve useful
tested behavior while moving it behind the required service/provider boundary.

### Deliverables

- [x] Pinned local Qwen3 provider with process-wide model reuse, lifecycle, health, models,
  and structured observability.
- [x] `POST /api/v1/simplify` returning Clinical, General Public, and Child-Friendly
  versions; deprecated `/api/v1/simplifications` remains as a compatibility operation.
- [x] External versioned structured prompt, entity/optional linked-concept grounding,
  deterministic generation, source-fact safety validation, typed errors, tests, Swagger,
  and the `/simplify` engineering dashboard.
- [ ] Representative clinical faithfulness/readability benchmark and acceptance thresholds.
- [ ] Persistence and shared-cache activation through the separately frozen Phase 3/4
  deployment boundary.

### Acceptance criteria

- Output preserves source facts, negation, uncertainty, values, units, doses, chronology,
  and evidence links while meeting recorded readability thresholds.
- Prompt injection and unsupported clinical instructions are handled safely.
- Model/prompt/retrieval versions, seed, parameters, confidence method, timing, and cache
  state are exposed and persisted.
- No deterministic test fallback can be enabled in production wiring.

Production validation on 2026-08-06 proved the immutable local checkpoint initializes and
generates all three levels on CPU. A negated synthetic report passed the grounding gate;
an HbA1c report was correctly rejected because the model introduced an unsupported numeric
detail. Phase 9 must remain in progress until representative numeric/laboratory reports
meet clinical acceptance without weakening the fail-closed policy.

## Phase 10 - Medical Verification

Status: `TECHNICALLY VERIFIED; LICENSE PENDING; PRODUCTION APPROVAL PENDING`

### Deliverables

- [x] Local-only PubMedBERT MedNLI verification provider and versioned
  /api/v1/verification resource.
- [x] Explicit checkpoint label mapping: contradiction=0, entailment=1, neutral=2; the
  service validates config.json and never infers or reorders labels.
- [x] Deterministic checks for numerical values, doses, units, percentages, dates,
  medication frequency, negation, and laterality; BLOCKED and REVIEW outputs stop
  Translation in the internal MVP pipeline.
- [ ] Verify the candidate license and obtain explicit production approval.
- Claim decomposition, source-evidence comparison, contradiction/neutral/entailment
  results, calibrated confidence, manual-review policy, persistence, and tests.

### Acceptance criteria

- Every patient-facing claim has evidence and a verification disposition.
- Contradicted, unsupported, and low-confidence claims cannot silently pass.
- Verification never silently rewrites content; revisions create a new traceable attempt.
- Clinical safety benchmarks and failure-path tests meet recorded thresholds.

## Phase 11 - Translation

Status: `IN PROGRESS - MVP; APPROVED LOCAL ARTIFACT AND E2E RUNTIME VALIDATED`

### Deliverables

- [x] IndicTrans2 provider and `/api/v1/translations` resource.
- [x] Pinned local-only `ai4bharat/indictrans2-en-indic-dist-200M` revision
  `173b94239f7c38886b2747b8d4a5db771a7e1232`, CUDA/CPU `auto` device selection, model
  reuse, batch inference, metadata, health, and fail-closed configuration. The approved
  local `model.safetensors` SHA-256 is verified before initialization.
- [x] Synthetic Hindi, Tamil, and Kannada runtime benchmark with exact protected
  numerical/unit/date values retained; evidence is in
  `New_current/benchmarks/translation/artifacts/2026-08-09-approved/`.
- [x] Real local OCR -> NER -> Simplification -> Translation transport run, using only real
  local models and a de-identified input. All three simplification levels reached the
  checksum-verified translation provider.
- Supported-language registry, terminology protection, back-check/quality policy, cache,
  persistence, confidence, model metadata, and tests.

### Acceptance criteria

- Translation preserves entities, negation, values, units, doses, warnings, formatting,
  and evidence references.
- Unsupported language and low-confidence output return explicit states.
- Representative medical quality and latency thresholds pass for every enabled language.
- Clinical validation, shared cache/persistence, and durable Phase 13 orchestration remain
  open; Phase 11 is not complete. The numeric-report simplification safety rejection is
  preserved and remains an upstream Phase 9 clinical-validation gate.

## Phase 12 - Text-to-Speech

Status: `DEFERRED FOR MVP`

### Deliverables

- Kokoro TTS provider and `/api/v1/speech` resource.
- Voice/dialect validation, safe pronunciation controls, object-storage output, database
  metadata, background jobs, confidence/quality method, deletion, and tests.

### Acceptance criteria

- Medical terms, numbers, units, and medication doses are audibly correct on the benchmark.
- Unsupported voice/language pairs and generation failures are explicit.
- Audio access is authorized, time-limited, traceable, and deleted by retention policy.

## Phase 13 - End-to-End Pipeline

Status: `IN PROGRESS - MVP PATIENT WORKFLOW`

### Deliverables

- Durable orchestrator connecting all approved stages through typed contracts.
- Stage state machine, checkpoints, resumability, cancellation, partial-result policy,
  provenance graph, unified patient response, API, UI integration, and tests.

### Acceptance criteria

- Representative reports complete from upload through verified optional translation and
  speech without manual database or queue intervention.
- Every output traces to source evidence, stage attempts, models, versions, and timings.
- Restart, retry, cancellation, duplicate, partial failure, low confidence, and manual
  review scenarios are verified without duplicated side effects.

## Phase 14 - Benchmarking and Production Hardening

An offline clinical-performance harness and empty, versioned de-identified dataset schema
exist under New_current/benchmarks/clinical_performance. It does not satisfy Phase 14
acceptance criteria: reviewed representative documents, gold labels, candidate artifacts,
clinical quality thresholds, and deployment validation remain pending.

Status: `PLANNED`

### Deliverables

- Versioned end-to-end benchmark suite and approved quality/safety thresholds.
- Load, soak, concurrency, GPU utilization, memory, cache, queue, and cost measurements.
- Threat model, privacy/security review, dependency and container scanning, observability,
  dashboards, alerts, backup/restore, disaster recovery, runbooks, and deployment gates.

### Acceptance criteria

- Accuracy, faithfulness, clinical safety, readability, latency, throughput, availability,
  and resource objectives pass on representative de-identified data.
- No critical security or privacy findings remain open.
- Backup/restore, rollback, model rollback, incident response, and retention/deletion are
  exercised successfully.
- Production release is explicitly approved; benchmark evidence and known limitations are
  recorded in `IMPLEMENTATION_LOG.md`.

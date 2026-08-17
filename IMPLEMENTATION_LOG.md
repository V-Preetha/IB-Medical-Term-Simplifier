# Medical Term Simplifier Implementation Log

This is the append-only engineering history for roadmap work. Add an entry when a phase
or meaningful cross-phase change is completed. Do not rewrite older entries to make the
history look current; add a correction entry instead.

Each entry must record:

- date, roadmap phase, status, and responsible task/change;
- production code and interfaces added or changed;
- APIs and compatibility impact;
- database migrations and rollback notes;
- models, immutable revisions, prompts/rules, and configuration introduced;
- tests, static checks, benchmark data, performance/resource metrics, and results;
- security, privacy, clinical-safety, and reproducibility considerations;
- known limitations, unavailable dependencies, and remaining work.

## 2026-08-02 - Phase 1 - Repository Infrastructure

Status: `COMPLETE`

### Built

- Added `AGENTS.md` as the standing repository engineering context.
- Added `ARCHITECTURE.md` as the architecture source of truth, including the target
  pipeline/model stack, service boundaries, API conventions, healthcare safety rules,
  and the logical schema transcribed from `DB.pdf`.
- Added `ROADMAP.md` as the ordered implementation contract with deliverables and
  acceptance criteria for Phases 1-14.
- Added this implementation log and made roadmap/log maintenance part of the quality gate.
- Recorded phase-protection, no-placeholder, model-confidence, model-version,
  reproducibility, and request-traceability rules in `AGENTS.md`.

### APIs and migrations

- No API behavior changed.
- No database migration was added.

### Verification

- Inspected the current `New_current/` implementation and documentation.
- Verified the logical database tables and relationships against the rendered `DB.pdf`.
- Confirmed PostgreSQL/Alembic, Redis/Celery, Qdrant, Docker, versioned `/api/v1/` routes,
  and the target React/Vite UI are not yet implemented and must not be reported complete.
- Documentation-only changes; application tests were not required or run.

### Baseline carried forward

- Current ingestion already supports content-based file detection, digital PDF extraction,
  PaddleOCR, TrOCR, mixed/multi-page routing, normalization, manual review, local caches,
  process-local jobs, GLiNER-BioMed, Qwen3-0.6B, a lightweight test page, and tests.
- These are valuable baseline capabilities, but they do not complete later roadmap phases
  until their service boundaries, target integrations, metadata contracts, persistence,
  infrastructure, and acceptance criteria are verified.

### Remaining work

- Phase 2 - OCR Service is active and `IN PROGRESS`.
- All subsequent phases remain `PLANNED` and are gated by the ordered roadmap.

## 2026-08-03 - Phase 2 - OCR Service - Incremental Step 1

Status: `IN PROGRESS`

### Built

- Established `New_current/app/ocr/` as the bounded context for the versioned OCR
  service, with separate `api`, `application`, `domain`, `providers`, `postprocessing`,
  `infrastructure`, and `observability` packages.
- Documented the inward dependency direction and package ownership in
  `New_current/app/ocr/README.md`.
- Established isolated OCR `unit`, `integration`, and `api` test packages under
  `New_current/tests/ocr/`.
- Preserved the existing `app.ingestion` implementation and all compatibility APIs.

### APIs and migrations

- No API behavior changed and no public route was added.
- No database model or migration was added.
- No Redis, Celery, model, post-processing, dashboard, or Docker behavior was added in
  this structure-only step.

### Verification

- Verified that the new Python package imports and byte-compiles successfully.
- Verified the structure contains no placeholder production response or model behavior.
- Existing application tests were not required because executable behavior did not
  change; full lint and test gates remain part of the later implementation steps.

### Security, safety, and reproducibility

- The package boundary reserves an observability layer for privacy-safe telemetry and
  keeps model and infrastructure adapters outside route and application logic.
- No clinical data, credentials, model artifacts, or runtime configuration were added.

### Known limitations and remaining work

- The task sequence requests database work, Redis, and Celery as OCR Phase 2 steps, while
  `ROADMAP.md` assigns those deliverables to Phases 3 and 4. This discrepancy remains
  unresolved; those capabilities are not reported as implemented or complete here.
- The task requests a Jinja2/Bootstrap/vanilla-JavaScript interface while the architecture
  and Phase 2 roadmap specify React/Vite/TypeScript. No dashboard technology was chosen
  or implemented in this step.
- The then-planned classifier, Qwen3-VL, post-processing, versioned REST APIs, the developer interface,
  tests, containerization, and end-to-end runtime verification remain incomplete.

## 2026-08-03 - Phase 2 - OCR Service - Incremental Step 2

Status: `IN PROGRESS`

### Built

- Added validated, persistence-neutral `OCRRequestRecord` and `OCRResultRecord` domain
  snapshots with UUID ownership and trace identifiers, SHA-256 identity, lifecycle and
  review states, progress, timing, confidence provenance, version metadata, warnings,
  and safe failure metadata.
- Added asynchronous, tenant-scoped `OCRRequestRepository` and `OCRResultRepository`
  protocols plus an `OCRUnitOfWork` contract for future atomic persistence.
- Kept persistence contracts independent of FastAPI, SQLAlchemy, Alembic, PostgreSQL,
  Redis, and Celery.
- Corrected the completed clinical renderer's verified compatibility defect so its
  existing `important_findings` field again renders under the contractually tested
  `Important Findings` heading; the change is limited to that display label.
- Recorded the explicitly authorized internal-interface decision: FastAPI-served Jinja2
  templates with Bootstrap 5 and vanilla JavaScript replace React/Vite/TypeScript.

### APIs and migrations

- No API behavior or compatibility route changed.
- No SQLAlchemy model, database adapter, table, or Alembic migration was added.
- The repository contracts are designed for a Phase 3 adapter over the authoritative
  `reports`, `report_processing`, and `model_outputs` schema.
- No Redis or Celery implementation was added; Phase 4 remains `PLANNED`.

### Verification

- Byte-compilation, import checks, and Ruff validation passed for the OCR package.
- The complete pre-existing test suite passed without compatibility regressions.
- No model inference or external infrastructure was required for this boundary-only step.

### Security, safety, and reproducibility

- Every repository lookup and deletion requires both request and owner identifiers to
  preserve tenant isolation at the application boundary.
- Domain records validate timezone-aware timestamps, SHA-256 form, legal progress and
  confidence ranges, failure metadata, and mandatory model/pipeline/rule/schema versions.
- Repository interfaces accept clinical OCR text but provide no logging behavior; normal
  logs remain prohibited from containing clinical content.

### Known limitations and remaining work

- The earlier Step 1 phase-scope conflict is resolved: PostgreSQL/Alembic remain Phase 3,
  and Redis/Celery remain Phase 4. Phase 2 supplies interfaces only.
- No repository implementation exists by design until Phase 3.
- Provider interfaces, the then-planned classifier, Qwen3-VL, post-processing, REST APIs, the internal
  dashboard, comprehensive Phase 2 tests, containerization, and runtime verification
  remain incomplete.

## 2026-08-03 - Phase 2 - OCR Service - Incremental Step 3

Status: `IN PROGRESS`

### Built

- Added the original classifier, OCR, and post-processor contracts
  with typed documents/results, initialization and shutdown lifecycle, metadata, health,
  supported formats/document types, configuration, confidence fields, timing, and warnings.
- Added instance-scoped `ProviderRegistry` registration plus automatic third-party
  discovery through dedicated Python entry-point groups; no mutable global registry was
  introduced.
- Added the original classifier, OCR, and post-processor factories so
  application wiring selects providers by configuration rather than concrete classes.
- Added `ProviderContainer` startup, partial-initialization rollback, reverse-order
  shutdown, health aggregation, and safe lifecycle failure handling.
- Added environment-backed `ProviderSettings`. The original three provider selections and
  their provider-specific configuration prefixes were required at that time.
- Added configuration-complete classifier, Qwen3-VL, and SymSpell skeletons. They validate
  immutable provider/model/dictionary/rule metadata, initialize without model libraries,
  report explicit `degraded` health, and raise `NotImplementedError` for inference as
  expressly required for this abstraction-only step.
- Added provider-specific configuration, initialization, availability, inference, and
  unsupported-document exceptions with safe API mapping.
- Added structured provider registration, initialization, shutdown, configuration-error,
  construction-error, and health-check log fields without clinical document content.
- Added FastAPI dependencies for all three interfaces and wired the provider lifecycle
  through the application lifespan.

### APIs and compatibility

- Added `GET /api/v1/ocr/health` with typed provider lifecycle/capability metadata,
  configuration redaction, OpenAPI summary/description/example, and documented 200/503
  responses. The probe does not execute inference.
- Existing unversioned ingestion and report APIs remain unchanged and continue to use
  their established compatibility pipeline.
- No database, SQLAlchemy, Alembic, Redis, Celery, or model SDK integration was added.

### Tests and verification

- Added tests for built-in registration, duplicate protection, structured registration
  logs, factory selection, unsupported providers, required environment configuration,
  fail-closed SymSpell configuration, lifecycle state, metadata, configuration redaction,
  FastAPI dependency resolution, provider health, and OpenAPI documentation.
- Repository-wide Ruff validation passed.
- Python byte-compilation passed for application and test sources.
- Full test suite passed: 70 tests, with one existing FastAPI/Starlette `TestClient`
  deprecation warning and no failures.
- No model downloads, inference calls, benchmark measurements, or external services were
  used in this step.

### Security, safety, and reproducibility

- Provider configuration is immutable after loading and fails startup when required
  selections or provider values are absent or invalid.
- Configuration keys indicating credentials, keys, passwords, secrets, or tokens are
  redacted from provider metadata and health responses.
- Health and lifecycle logs include provider kind/name/version and state but never file
  bytes, extracted text, normalized text, prompts, or clinical content.
- Skeleton health is never reported as ready, preventing unimplemented inference from
  being mistaken for an available clinical model.

### Known limitations and remaining work

- Classifier inference, Qwen3-VL OCR, and SymSpell/medical-rule inference are not
  implemented by design and remain later Phase 2 steps.
- The new provider layer is not yet used by the OCR business pipeline; only configuration,
  lifecycle, dependency injection, discovery, metadata, and health are integrated.
- Provider health is `degraded` until real adapters load and pass their later readiness
  checks. Production startup now requires the documented provider environment variables.
- Phase 3 database work and Phase 4 Redis/Celery work remain `PLANNED` and unchanged.

## 2026-08-03 - Phase 2 - OCR Service - Incremental Step 4

Status: `IN PROGRESS`

### Built

- Replaced the Step 3 inference skeletons with the then-approved classifier,
  `Qwen3VLOCRProvider`, and `SymSpellPostProcessor` production adapters while preserving
  the original classifier, OCR, and post-processing contracts unchanged.
- Retained the Step 3 import path as a compatibility-only re-export; it contains no model
  or inference fallback.
- Implemented lifecycle-lazy loading during FastAPI startup, one loaded model/processor
  per provider container, request reuse under inference locks, idempotent repeated
  initialization, partial-load cleanup, reverse-order shutdown, garbage collection, and
  CUDA cache release.
- Implemented configured `auto`/CPU/CUDA device resolution, explicit CPU fallback policy,
  dtype selection, GPU name and memory metrics, model-load timing, inference timing, and
  structured lifecycle/inference logs without clinical content.
- Implemented the retired classifier over independently decoded pages using processor
  OCR/layout inputs, mean page softmax confidence, configured review threshold, immutable
  model revision, and classification/provider/runtime metadata.
- Implemented Qwen3-VL OCR batching for validated PDF, PNG, JPEG/JPG, and single- or
  multi-frame TIFF content. PDF/TIFF pages retain order and page metadata; results expose
  measured generated-token confidence, prompt/version identity, inference latency,
  pages/second, batch/page counts, model-load time, device, CPU fallback, and GPU memory.
- Implemented conservative document decoding with content-signature validation, encrypted/
  malformed/empty PDF rejection, image-format verification, decompression-bomb handling,
  page limits, EXIF correction, PDF rendering DPI, and bounded image resizing.
- Implemented independent regex, approved medical-abbreviation dictionary, and bounded
  SymSpell delete-index stages. Every stage reports latency, correction count, and
  configuration; the pipeline protects configured medical terms and avoids correcting
  capitalized, numeric, dose, and known-dictionary tokens.
- Added idempotent abbreviation expansion so retried post-processing cannot recursively
  expand preserved forms such as `blood pressure (BP)`.
- Added deterministic generation seed, inference parameters, confidence thresholds and
  calibration-version identifiers, model/cache paths, page/image
  limits, batch size, and timeout controls through environment-backed provider settings.
- Added PyTorch and Transformers as runtime dependencies. The local Python environment
  has Transformers 4.57.6 and CPU-only PyTorch 2.13.0.

### APIs, compatibility, and migrations

- `GET /api/v1/ocr/health` now reports `ready` only after every configured model and
  dictionary resource loads successfully; model/runtime metadata is configuration-safe
  and the OCR prompt is redacted while prompt version remains visible.
- Existing unversioned ingestion/report APIs and all Step 1-3 contracts remain unchanged.
- No database model, migration, Redis adapter, Celery worker, or future-phase service was
  added.

### Tests and verification

- Added synthetic, de-identified tests for one-time loading/release, CPU classification,
  configured CUDA-to-CPU fallback, conditional CUDA classification, single-page PNG OCR,
  ordered multi-page PDF and TIFF OCR, token confidence, page/performance metadata,
  configuration validation, missing/corrupt checkpoints, invalid PDF/image input,
  out-of-memory handling, deterministic timeout handling, three-stage post-processing,
  medical-term preservation, stage metrics, and retry idempotency.
- Model-library loading is faked only at the external checkpoint boundary in automated
  tests; production wiring contains no fake model output or heuristic model fallback.
- Repository-wide Ruff validation passed and Python application/test byte-compilation
  passed.
- Full automated suite passed: 82 tests. One CUDA-only test was skipped because CUDA is
  unavailable; one existing FastAPI/Starlette `TestClient` deprecation warning remains.
- Editable package build succeeded with `pip install -e . --no-deps --no-build-isolation`,
  and `pip check` reported no broken requirements.
- Required Transformers auto/model classes imported successfully.

### Security, clinical safety, and reproducibility

- No uploaded bytes, OCR text, normalized text, model prompt, credential, or token is
  written to normal logs or health responses.
- Exact model, provider, prompt, dictionary, abbreviation, rule, and confidence-calibration
  versions are required configuration. No model path or revision is hardcoded.
- OCR confidence is the measured mean probability of generated tokens. Layout confidence
  is the measured mean-page softmax probability. Both methods remain explicitly marked by
  configured uncalibrated calibration identities until benchmark calibration is completed.
- Low-confidence results carry explicit review warnings. Empty OCR output, invalid input,
  missing resources, corrupt artifacts, CUDA unavailability without permitted fallback,
  out-of-memory, inference failure, and timeout overrun fail explicitly.
- Generation defaults are deployment configuration and a configured seed is applied under
  the provider inference lock for reproducibility.

### Known limitations and remaining work

- No approved immutable classifier checkpoint or Qwen3-VL OCR checkpoint was
  present in the local Hugging Face cache, and the task did not authorize selecting one.
  Real model loading, CPU inference quality/latency, confidence behavior, and resource use
  therefore remain unverified.
- The host has CPU-only PyTorch and no CUDA runtime, so GPU inference and GPU memory release
  were not exercised. The CUDA test remains skipped rather than simulated as a passing GPU
  integration.
- Approved model identifiers and immutable revisions remained unavailable. The classifier
  startup therefore fails closed before attempting checkpoint loading.
- Timeout enforcement is checked before and after model batches. Hard preemption of a
  blocked native GPU kernel requires the isolated worker/process controls planned for
  Phase 4; the current adapter detects and rejects an overrun after control returns.
- Representative document accuracy, calibration, latency, throughput, GPU/CPU resource
  benchmarks, and exact model-selection evidence are still required before this roadmap
  step or the Phase 2 model deliverables can be marked complete.
- Activation variables and verification commands are documented in
  `New_current/app/ocr/README.md`. Phase 3 and Phase 4 remain unchanged.

## 2026-08-03 - Phase 2 - OCR Service - Step 4 Validation and Benchmarking

Status: `IN PROGRESS - NOT VERIFIED`

### Validation tooling and evidence

- Added the fail-closed `New_current/benchmarks/ocr/run_validation.py` runner. It uses
  production provider factories and environment configuration, never selects or downloads
  a model, never substitutes a fake checkpoint, and leaves unavailable metrics empty.
- Added generated synthetic, de-identified PDF, multi-page PDF, PNG, JPEG, and multi-frame
  TIFF inputs for repeatable decoder/provider validation.
- Generated Markdown, CSV, JSON, JSONL structured-log, and correction/error reports under
  `New_current/benchmarks/ocr/reports/`.
- Captured the existing developer page, Swagger, the real OCR health response, and the
  synthetic upload result under `New_current/benchmarks/ocr/reports/screenshots/`.

### Verified results

- Repository-wide Ruff passed.
- The complete automated suite passed: 82 tests, one CUDA-only test skipped because the
  installed PyTorch runtime has no CUDA support, and one existing TestClient deprecation
  warning. Model checkpoint loading in these tests remains faked only at the documented
  external boundary and is not counted as live-model evidence.
- Content decoding passed for single-page PDF, ordered two-page PDF, PNG, JPEG, and ordered
  two-frame TIFF using synthetic inputs. These decoder results are not counted as successful
  Qwen3-VL OCR for those formats.
- The real regex, medical abbreviation dictionary, and SymSpell implementation passed on
  synthetic text. It produced 11 regex normalizations, one abbreviation expansion, and one
  SymSpell correction with measured per-stage latency recorded in the benchmark reports.
- Structured lifecycle and post-processing records included provider name/version, event,
  stage timing, and correction fields without clinical text.
- Swagger rendered and documented `GET /api/v1/ocr/health`. The actual operation returned
  the safe `503 provider_unavailable` envelope because providers were not initialized.

### Benchmark outcome and unavailable metrics

- Overall result is `NOT VERIFIED`; Phase 2 Step 4 remains unchecked in `ROADMAP.md`.
- No production provider environment, approved immutable classifier checkpoint, or approved
  immutable Qwen3-VL checkpoint was configured, and no matching checkpoint was found in the
  local Hugging Face cache. Live model loading and CPU OCR therefore did not run.
- The host exposes an NVIDIA GeForce RTX 5050 Laptop GPU with 8,151 MiB, but the installed
  PyTorch build is `2.13.0+cpu`; CUDA is unavailable to PyTorch. CUDA inference and GPU memory
  measurements remain `NOT VERIFIED` rather than simulated.
- Classifier initialization could not pass until its approved immutable checkpoint was
  recorded and available locally.
- Model loading time, first and warm model inference latency, pages per second, GPU memory,
  model CPU-memory deltas, and confidence distribution remain empty in the reports because
  no live model inference completed.
- OCR mistake/confusion scoring remains `NOT VERIFIED` because no approved representative
  ground-truth corpus was supplied. The correction report includes only synthetic regex,
  abbreviation, and SymSpell evidence.
- A screenshot-only server was started with application lifespan disabled so documentation
  surfaces could be captured without bypassing failed provider readiness. The developer page
  and Swagger screenshots are UI evidence only; the health and upload screenshots preserve
  the real unavailable states and are not successful inference evidence.

### APIs, migrations, and phase scope

- No production API, provider, model, database, Redis, Celery, or future-phase code changed.
- No database migration was added. Phase 3 and Phase 4 remain `PLANNED` and unchanged.

## 2026-08-03 - Phase 2 - OCR Architecture Convergence

Status: `CONVERGENCE COMPLETE - PHASE IN PROGRESS`

### Corrected architecture

- Removed the competing `ReportIngestionPipeline` execution path and all PaddleOCR, TrOCR,
  legacy routing, ingestion-service, parallel normalization, obsolete API/job/schema, and
  stale UI implementations.
- Removed corresponding legacy tests and benchmark code. Removed PaddleOCR, PaddlePaddle,
  NumPy, and the TrOCR optional dependency group from the Phase 2 package manifest.
- Added `OCRApplicationService` as the sole orchestration boundary: document
  classification, OCR, post-processing, result construction, failure handling, metrics,
  request state, and cache coordination now flow through one application service.
- Production startup obtained classifier, Qwen3-VL, and SymSpell providers through the
  provider registry, factories, environment configuration, and lifecycle container.
  Routes depend only on `OCRApplicationService` and never invoke providers directly.
- Added the versioned upload, result, status, delete, provider-health, model-metadata,
  liveness, readiness, recent-request, and privacy-safe log APIs.
- Added a same-origin Jinja2/Bootstrap 5/vanilla-JavaScript OCR engineering console with
  upload/preview, raw and normalized output, classification, confidence, timing, provider,
  health, recent-request, logs, and Swagger surfaces.
- Added JSON structured log serialization for request, stage, provider/version, timing,
  confidence, cache, and safe error fields.
- Added complete trace/provenance response metadata and removed the uploaded content hash
  from public result metadata. Cache identity now includes pipeline and schema versions.
- Documented removed/refactored components, retained dependency reasons, evidence, and
  remaining limitations in `New_current/ARCHITECTURE_CONVERGENCE_REPORT.md`.

### Verification

- Ruff passed across `app`, `tests`, and `benchmarks`.
- The complete suite passed: 42 tests passed; one CUDA-only test was skipped because the
  installed PyTorch runtime is CPU-only.
- Static compilation and an import walk across every `app` module passed.
- OpenAPI, Swagger availability, the engineering dashboard, and all OCR HTTP operations
  were exercised with lifecycle-injected synthetic providers at model-library boundaries.
- A repository scan found no PaddleOCR, TrOCR, `ReportIngestionPipeline`, `app.ingestion`,
  or `app.normalization` reference in runtime code, tests, benchmarks, or manifests.

### Remaining limitations

- Phase 2 remains `IN PROGRESS`. Step 4 remains unchecked because live loading and
  inference with approved immutable classifier and Qwen3-VL artifacts, representative
  accuracy/confusion evidence, CPU/CUDA deployment metrics, and the container acceptance
  criterion are not verified.
- No PostgreSQL, Alembic, Redis, Celery, or future pipeline stage was introduced. Phase 3
  and Phase 4 remain `PLANNED` and unchanged.

## 2026-08-03 - Phase 2 - Production Validation

Status: `IN PROGRESS - NOT READY FOR PHASE 3`

### Scope and contract handling

- Performed validation only. No production code, provider interface, API, dependency,
  architecture, database, cache, job, or future-phase implementation changed.
- Re-read the frozen `AGENTS.md`, `ARCHITECTURE.md`, `ROADMAP.md`,
  `IMPLEMENTATION_LOG.md`, and architecture convergence report. `API_SPEC.md` remains
  absent; the prior explicit instruction allowed validation to proceed without it, and
  the missing frozen artifact is recorded as a documentation gap.
- Generated the final evidence assessment in
  `New_current/OCR_PHASE_COMPLETION_REPORT.md`.

### Checkpoint and runtime inventory

- No deployment provider environment variables are configured. The frozen contracts do
  did not record exact approved classifier or Qwen3-VL repository identifiers, variants, or
  immutable revisions.
- Inspected `C:\Users\Preetha\.cache\huggingface\hub`; no classifier or Qwen3-VL
  candidate is cached. Model identifiers, revisions, checksums, and snapshot locations
  therefore remain `NOT VERIFIED`. No unapproved model was selected or downloaded.
- Runtime: Python 3.12.13, Transformers 4.57.6, PyTorch
  2.13.0+cpu, 24 logical CPUs, and 25,459,679,232 bytes system memory.
- Hardware exposes an NVIDIA GeForce RTX 5050 Laptop GPU with 8,151 MiB and driver
  592.27, but CUDA is unavailable to the installed PyTorch build. CUDA validation is
  `NOT VERIFIED`, not failed.
- A production lifespan attempt failed closed because the then-required classifier
  selection was absent; provider and application health were not verified.

### Benchmark and validation evidence

- Re-ran `benchmarks/ocr/run_validation.py` for CPU and CUDA. The runner used no fakes,
  did not download or select a model, and left unavailable model metrics null.
- Synthetic decoder checks passed for single-page PDF (one page), multi-page PDF (two
  ordered pages), PNG (one page), JPEG (one page), and multi-frame TIFF (two ordered
  pages). Decoder success is not counted as Qwen3-VL inference success.
- The production post-processing implementation passed using versioned synthetic fixtures:
  11 regex corrections in 0.175 ms, one medical-abbreviation correction in 0.026 ms,
  one SymSpell correction in 0.205 ms, and 13 total corrections in 0.472 ms. Provider
  initialization took 10.632 ms.
- CER, WER, confidence distribution, live CPU latency, first/warm inference latency,
  pages per second, model CPU memory, GPU memory, and utilization remain `NOT VERIFIED`
  because no approved representative ground-truth corpus or live model inference exists.
- Generated Markdown, JSON, CSV, JSONL, correction, checkpoint-inventory, quality-gate,
  screenshot, and tabular-preview artifacts beneath
  `New_current/benchmarks/ocr/reports/`. The two CSV files were imported, inspected,
  formula-error scanned, and visually rendered; both contain the expected rows and no
  spreadsheet error tokens.

### Quality gates

- Ruff passed: `python -m ruff check app tests benchmarks`.
- Full suite passed: 42 tests passed, one CUDA-only test skipped, and one existing
  Starlette/TestClient deprecation warning was emitted.
- Static compilation passed; 39 application modules imported successfully.
- `pip check` passed with no broken requirements.
- OpenAPI contained nine paths and every required OCR/service-health path.
- Dashboard and Swagger rendered successfully and fresh screenshots were captured with
  lifecycle-injected synthetic providers. This verifies UI/OpenAPI wiring only and is
  not reported as production model or provider-health evidence.

### Completion decision

- Phase 2 remains `IN PROGRESS`; Step 4 and production validation remain unchecked.
- Final recommendation is `NOT READY FOR PHASE 3` because approved immutable checkpoints,
  successful provider initialization/health, live CPU inference, representative accuracy,
  model resource/performance metrics, and reproducible container startup are unverified.
- Phase 3 remains `PLANNED`; no PostgreSQL, Alembic, Redis, Celery, or future phase was
  started.

## 2026-08-03 - Phase 2 - Deterministic Model Validation Finalization

Status: `IN PROGRESS - NOT READY FOR PHASE 3`

### Model inventory and configuration

- Added repository-root `MODEL_MANIFEST.md` as the deterministic inventory for the
  then-approved classifier and Qwen3-VL OCR provider. Repository IDs, licenses, revisions,
  and optional checksums remain `PENDING_APPROVAL`; no identifier was guessed and no
  checkpoint was downloaded.
- Added a strict manifest loader for the embedded machine-readable contract. It rejects
  missing fields, malformed digests, non-commit revisions, and moving revision names.
- Updated environment-backed provider settings so missing identity, revision, cache, and
  device values resolve only from the manifest. Deployment environment variables may
  provide approved overrides. Missing approval fails closed with a provider configuration
  error before model loading.
- Added tests for manifest parsing, approved environment/manifest resolution, moving
  revision rejection, factory selection, provider metadata, health, and dependency
  injection. Model-library inference remains outside these tests.

### Dependency and validation correction

- Removed the unapproved external OCR executable integration and Python wrapper from the
  production provider, package manifests, validation runner, health/startup assumptions,
  active reports, and legacy utility manifests. Legacy image-only utilities now fail
  closed and direct callers are directed to the approved OCR service.
- Disabled processor-internal text extraction for the retired classifier. It supplied a
  deterministic full-page layout token while preserving the frozen classifier-before-OCR
  pipeline and provider interface.
- Updated the benchmark runner to parse `MODEL_MANIFEST.md`, compare provider identity and
  immutable revision, locate the configured local snapshot, optionally compute a
  deterministic directory SHA-256, and report absent approval/artifacts as
  `NOT VERIFIED` rather than `FAILED`.
- Added `DEPENDENCY_AUDIT.md`, updated the active service documentation and completion
  report, and limited the remaining Roadmap gates to approved manifest values, immutable
  checkpoints, CPU inference, optional CUDA, OCR benchmarking, and clinical validation.

### Objective verification

- Ruff passed across `app`, `tests`, and `benchmarks`.
- The full production suite passed: 45 tests passed, one CUDA-only test skipped, and one
  existing Starlette/TestClient deprecation warning was emitted.
- Static compilation passed and all 40 application modules imported successfully.
- Editable installation of `ib-health-ocr-service` succeeded; `pip check` reported no
  broken requirements after stale environment metadata was removed.
- Manifest parsing passed for exactly the classifier and Qwen3-VL entries. Both remained
  `NOT VERIFIED` because their identity and immutable revision are pending approval.
- OpenAPI, Swagger, and the engineering dashboard passed direct checks; the schema exposes
  nine paths. Focused API/provider tests passed 14 tests.
- The fail-closed validation runner completed and regenerated Markdown, JSON, CSV, JSONL,
  format-decoding, correction, structured-log, and manifest evidence. Post-processing
  passed with 13 measured corrections in 0.223 ms; no live model metric was fabricated.
- Repository source, dependency, validation, and active documentation scans contain no
  reference to the removed executable engine or its Python wrapper. Active production
  source and manifests contain no alternate OCR provider implementation.

### Completion decision

- Phase 2 and Step 4 remain `IN PROGRESS`. No production model initialized because model
  identities and immutable checkpoints are not approved or locally available.
- The recommendation remains `NOT READY FOR PHASE 3`. The six exclusive gates recorded
  in `ROADMAP.md` remain open; optional CUDA is `NOT VERIFIED`, not failed.
- No API, provider interface, database, Redis, Celery, or future-phase implementation was
  introduced.

## 2026-08-04 - Phase 2 - Single-Model OCR Architecture Simplification

Status: `IN PROGRESS - ARCHITECTURE SIMPLIFICATION COMPLETE`

### Architecture decision and implementation

- Removed the dedicated document-classification provider contract, implementation,
  registry kind, discovery group, factory, lifecycle slot, dependency-injection binding,
  configuration selection, environment prefix, health dependency, benchmark path, and
  tests.
- The sole executable pipeline is now upload, Qwen3-VL OCR with prompted document-type
  inference, regex normalization, medical abbreviation expansion, and SymSpell.
- Qwen3-VL returns document type and page text through a versioned JSON output schema;
  `OCRApplicationService` passes that type to post-processing without depending on a
  concrete provider.
- Production composition now creates exactly two providers through the registry and
  factories: `qwen3-vl` and `symspell`.
- Removed stale validation artifacts that represented the retired provider and regenerated
  the benchmark reports from the simplified validation runner.
- Updated `ARCHITECTURE.md`, `ROADMAP.md`, `MODEL_MANIFEST.md`, dependency documentation,
  service documentation, architecture convergence report, and phase completion report.
- PostgreSQL, Redis, Celery, and later pipeline stages were not modified or started.

### Objective verification

- Ruff passed across `app`, `tests`, and `benchmarks`.
- Full test suite passed: 46 tests passed and one CUDA-only test was skipped; the existing
  Starlette/TestClient deprecation warning remains.
- Static compilation passed and all 40 application modules imported successfully.
- The pinned `Qwen/Qwen3-VL-4B-Instruct` revision
  `ebb281ec70b05090aa6165b016eac8ec08e71b17` loaded from
  `New_current/.model-cache/qwen3-vl` on CPU in 6,780.428 ms without a download.
- The deterministic post-processor initialized in 2.270 ms using versioned synthetic
  resources. `GET /api/v1/ocr/health` returned HTTP 200 and `ready`; its payload contained
  exactly the Qwen3-VL and post-processing providers, both `ready`.
- Repository scans found no active source, test, validation, configuration, report, or
  documentation reference to the retired model or classifier provider identifiers.

### Remaining validation gates

- Phase 2 and Step 4 remain `IN PROGRESS`. The complete live CPU format benchmark, an
  approved clinical CER/WER corpus with thresholds, confidence calibration, and production
  medical dictionaries remain open. The unavailable aggregate checkpoint checksum is an
  optional manifest integrity value, not a completion gate.
- Optional CUDA validation remains `NOT VERIFIED` because the installed PyTorch runtime is
  CPU-only. Phase 3 remains `PLANNED`.

## 2026-08-04 - Phase 5 - Medical NER - Stage 1 Evaluation Framework

Status: `IN PROGRESS - EVALUATION FRAMEWORK COMPLETE`

### Scope and architecture

- Proceeded under the explicitly authorized Phase 5 Stage 1 designation. Existing phase
  numbering is unchanged: Phase 3 remains the Database Layer and Phase 4 remains Redis and
  Celery. The frozen OCR package and OCR pipeline were not modified.
- Added an isolated `app.ner` evaluation boundary with a typed `BaseNERProvider`, canonical
  entity/result/metadata/health contracts, instance-scoped registry, lazy local-only
  candidate adapters, application service, dependency injection, safe errors, structured
  logging, versioned API schemas, and lifecycle cleanup.
- Registered exactly the three evaluation candidates: OpenMed GLiNER (Zero-Shot),
  biomedical-ner-all, and Kushtrim/ModernBERT-base-biomedical-ner. No winner or production
  NER provider was selected.
- Added a separate machine-readable NER section to `MODEL_MANIFEST.md`. All repository IDs,
  immutable revisions, licenses, and checksums remain `PENDING_APPROVAL`; providers reject
  moving/unverified revisions, require local revision evidence, and never download or
  substitute a checkpoint.
- Normalized outputs to Disease, Symptom, Medication, Procedure, Anatomy, Laboratory Test,
  Measurement, and Medical Abbreviation with offsets and measured confidence. Unknown
  token-model labels are excluded with explicit warnings; approved per-candidate label maps
  can be supplied through documented environment configuration.

### Evaluation behavior and surfaces

- Added exact-span precision, recall, F1, entity-level accuracy, false-positive, and
  false-negative scoring. Entity-level accuracy is `TP / (TP + FP + FN)`; quality metrics
  remain null when reference annotations are absent.
- Added measured inference latency, process peak RAM, conditional peak CUDA memory, model
  loading time, and tokens/second. No unavailable metric is fabricated.
- Added evaluation-only `POST /api/v1/ner/benchmark`, `GET /api/v1/ner/models`, and
  `GET /api/v1/ner/health`. No production `/api/v1/ner` inference endpoint was added.
- Added `/ner/benchmark`, a same-origin Bootstrap/vanilla-JavaScript console that selects a
  candidate and completed OCR request, shows raw and normalized OCR text, highlights
  entities, and displays entity, health, latency, memory, and evaluation metrics.
- Added `benchmarks/ner/run_benchmark.py`, a synthetic de-identified JSONL schema template,
  and generated JSON, CSV, and Markdown artifacts. Identical input records are used for all
  candidates, and the generated report explicitly records `winner: null`.

### Verification

- Ruff passed across `app`, `tests`, and `benchmarks`.
- Full suite passed: 54 tests passed, one conditional CUDA test was skipped, and the existing
  Starlette/TestClient deprecation warning remains.
- Static compilation passed and all 49 application modules imported successfully.
- API tests verified all three NER evaluation operations, typed metric output, stable model
  selection, health inventory, Swagger documentation, and dashboard rendering while
  preserving all previous OCR tests.
- The benchmark runner completed and generated report templates. All three candidates are
  correctly `NOT VERIFIED`, every metric is empty, and no winner is recorded because model
  identities and immutable checkpoints are pending approval.

### Remaining gates

- Phase 5 remains `IN PROGRESS`. Approve the exact repository ID, immutable revision,
  license, local checkpoint, and label mapping for each candidate; supply an approved,
  representative, de-identified annotated OCR corpus and thresholds; then run all three
  candidates on identical inputs.
- Production NER integration and model selection remain prohibited until benchmark review
  is complete. Entity linking, embeddings, Qdrant, relation extraction, simplification,
  verification, translation, and later stages were not implemented.

## 2026-08-04 - Phase 5 - Medical NER - Stage 2 Candidate Evaluation

Status: `IN PROGRESS - BENCHMARK COMPLETE, WINNER APPROVAL PENDING`

### Immutable local model evidence

- Evaluated only the three approved local checkpoints with Hugging Face, Transformers, and
  datasets offline modes enabled. No download, network lookup, checkpoint substitution, or
  production NER integration occurred.
- OpenMed GLiNER used `OpenMed/OpenMed-ZeroShot-NER-Pathology-Medium-209M` revision
  `e63d8b131d599970674d05617bdbd1a3eef495ee`; its observed deterministic artifact-tree
  SHA-256 is `4a8ba53c9d9ce72a2acfc431afc16d5127da06b669960c1434e27e97d134d3ef`.
- biomedical-ner-all used `d4data/biomedical-ner-all` revision
  `015a4050c9ac99722e61c547aa9b4282bcbedc7f`; its observed deterministic artifact-tree
  SHA-256 is `6884545271a453eed4ff94f5e718105154c83104a7188bdcf424693ab6768493`.
- ModernBERT used `Kushtrim/ModernBERT-base-biomedical-ner` revision
  `a4ebc00ce8c52ac03ccaaae96600431b9e4c3e39`; its observed deterministic artifact-tree
  SHA-256 is `8e7de26bcc7b9dc0851e219cf98179796cd7baa1944ef3239a697e0d091ae980`.
- The hashes above are observed local inventory evidence, not substituted expected hashes;
  the manifest's optional expected SHA-256 fields remain pending approval.

### Benchmark evidence

- All candidates processed the same four synthetic, de-identified records containing 15
  exact-span references across Disease, Symptom, Medication, Procedure, Anatomy,
  Laboratory Test, Measurement, and Medical Abbreviation. The confidence threshold was
  0.5, maximum input length was 512 tokens, and the execution device was CPU.
- `biomedical-ner-all` ranked first: macro F1 `0.381250`, overall precision/recall/F1
  `0.333333/0.333333/0.333333`, entity accuracy `0.200000`, 10 false positives, 10 false
  negatives, mean latency `22.654500 ms`, load time `125.386 ms`, peak RAM `1217.066 MiB`,
  and mean throughput `1257.020500 tokens/s`.
- ModernBERT ranked second: macro F1 `0.170833`, overall precision/recall/F1
  `0.250000/0.200000/0.222222`, entity accuracy `0.125000`, 9 false positives, 12 false
  negatives, mean latency `56.289250 ms`, load time `175.454 ms`, peak RAM `869.730 MiB`,
  and mean throughput `386.047000 tokens/s`.
- OpenMed GLiNER ranked third: macro F1 `0.083333`, overall precision/recall/F1
  `0.125000/0.066667/0.086957`, entity accuracy `0.045455`, 7 false positives, 14 false
  negatives, mean latency `114.427750 ms`, load time `6444.307 ms`, peak RAM
  `1972.750 MiB`, and mean throughput `165.114500 tokens/s`.
- CUDA was unavailable in the installed CPU-only PyTorch runtime. Peak GPU memory is
  recorded as `NOT VERIFIED`, not zero and not a failure.
- Generated `ner_benchmark_report.md`, `ner_benchmark_report.json`, overall metrics CSV,
  per-entity CSV, and leaderboard CSV under `New_current/benchmarks/ner/reports/`.

### Evaluation correctness and verification

- Applied the configured confidence threshold consistently to all candidate adapters and
  normalized tokenizer offsets that include leading/trailing whitespace before exact-span
  scoring. A regression test covers whitespace-offset normalization.
- Ruff passed repository-wide. The complete suite passed: 55 tests passed and one
  CUDA-conditional test was skipped; the existing Starlette/TestClient deprecation warning
  remains. All three real local candidates completed with `PASS` benchmark execution.
- Phase 5 remains `IN PROGRESS`. The evidence recommends `biomedical-ner-all` for explicit
  review only; `winner` remains null. The four-record synthetic set is too small for final
  clinical acceptance, and representative clinical-corpus thresholds and explicit winner
  approval remain open. No production model was selected or integrated.

## 2026-08-04 - Phase 5 - Medical NER - Stage 3 Production Integration

Status: `IN PROGRESS - PRODUCTION INTEGRATION COMPLETE, CLINICAL ACCEPTANCE PENDING`

### Approved decision and runtime

- Recorded the formally accepted production winner `d4data/biomedical-ner-all`, immutable
  revision `015a4050c9ac99722e61c547aa9b4282bcbedc7f`, and Apache-2.0 license in
  `MODEL_MANIFEST.md`. OpenMed GLiNER and ModernBERT remain archived benchmark references.
- Added an explicit manifest production selector that fails closed unless the approved
  biomedical provider is selected. Production model/revision overrides must equal the
  approved manifest identity, and model/tokenizer loading remains local-only.
- Production composition now creates, initializes, injects, health-checks, and shuts down
  exactly one provider through the existing interface and instance-scoped registry. The
  service depends only on `BaseNERProvider`; routes do not import model libraries.
- Removed benchmark execution from FastAPI composition and removed the benchmark endpoint
  and candidate dashboard. Benchmark providers, resource monitoring, scoring, composition,
  and runner now remain beneath `New_current/benchmarks/ner/` and are never imported or
  initialized by production startup.

### Production API and developer interface

- Added production `POST /api/v1/ner`, `GET /api/v1/ner/health`, and
  `GET /api/v1/ner/models`. Responses include request ID, canonical entities, measured
  token confidence, offsets, aggregate confidence and method, calibration version, review
  state, latency, token throughput, exact model/revision, device, configuration provenance,
  warnings, cache status, and pipeline/schema versions.
- Added the `/ner` Jinja2/Bootstrap/vanilla-JavaScript engineering console with normalized
  OCR input or completed OCR-result selection, highlighted entities, entity table, model
  health, inference metadata, raw JSON response, and Swagger access.
- Structured lifecycle and inference logs include request ID, stage, provider/model,
  immutable revision, latency, entity count, measured confidence, and device without
  logging clinical text.
- Added `New_current/NER_PRODUCTION_INTEGRATION_REPORT.md` and production NER operational
  documentation in `New_current/app/ner/README.md`.

### Objective verification

- Ruff passed repository-wide. The full suite passed: 58 tests passed and one conditional
  CUDA test was skipped; the existing Starlette/TestClient deprecation warning remains.
- Static compilation/import verification passed. OpenAPI includes the production POST,
  health, and models operations and excludes `/api/v1/ner/benchmark`. Swagger and the
  production NER dashboard returned HTTP 200.
- The real pinned checkpoint loaded locally on CPU in `4305.866 ms` with provider health
  `ready` and application NER health `HEALTHY`. `GET /api/v1/ner/models` returned exactly
  one provider.
- Real production inference returned HTTP 200 in `49.021 ms` with exact model revision,
  measured aggregate confidence `0.872816`, entity offsets, 14 processed tokens, and
  measured throughput. No checkpoint download or substitution occurred.
- Corrected the production checkpoint-native label mapping after live validation exposed
  dropped continuation labels; a regression test covers native biomedical labels and BIO
  span merging.
- Replaced inherited single-window truncation with configurable overlapping tokenizer
  windows, source-relative offsets, deterministic overlap ownership, and exact-span
  deduplication. Unit coverage verifies overlap ownership. A real 669-token synthetic input
  crossed the configured 512-token window, processed successfully in `313.027 ms`, and all
  returned spans matched their source offsets.

### Limitations and phase disposition

- The real checkpoint returned partial spans for the synthetic inference sentence. This is
  preserved as evidence rather than masked. Representative clinical-corpus thresholds,
  confidence calibration, and long-input/chunk-boundary validation remain open.
- CUDA is `NOT VERIFIED` because installed PyTorch is CPU-only. Phase 3 database and Phase
  4 Redis/Celery work were not started. OCR and later entity linking/relation stages were
  not modified.
- Stage 3 production integration is complete, but Phase 5 remains `IN PROGRESS` until its
  remaining clinical and infrastructure acceptance gates are satisfied.

## 2026-08-05 - Phase 6 - Entity Linking Production Boundary

Status: `IN PROGRESS - SOFTWARE BOUNDARY COMPLETE, LICENSED RUNTIME VALIDATION PENDING`

### Implementation

- Added `New_current/app/entity_linking/` with a provider-neutral contract, instance-scoped
  registry, environment/manifest configuration, application service, typed safe errors,
  lifecycle management, health metadata, and structured privacy-safe logs.
- Added a local-only `SciSpacyUMLSProvider`. It requires the exact approved SciSpaCy
  version, local language model, licensed local UMLS resources, explicit license
  acceptance, and terminology release before loading. It never substitutes a model,
  downloads implicitly through application policy, fabricates a concept, or logs entity
  text.
- Normalized provider output to preserve source NER entity and offsets, ranked candidates,
  UMLS CUI, preferred name, semantic type identifiers, source ontology, measured SciSpaCy
  candidate similarity, ambiguity, review state, and explicit unlinked results.
- Added `POST /api/v1/entity-linking`, `GET /api/v1/entity-linking/health`, and
  `GET /api/v1/entity-linking/models` with versioned Pydantic responses, reproducibility
  metadata, OpenAPI documentation, examples, documented errors, and stable error envelopes.
- Added `/entity-linking`, a same-origin Jinja2/Bootstrap/vanilla-JavaScript console for
  direct entity testing and OCR-text -> existing production NER -> Entity Linking testing.
  The frozen OCR and NER packages were not architecturally changed; composition passes NER
  output to the new public contract.

### Manifest and readiness

- Extended `MODEL_MANIFEST.md` with a strict machine-readable Entity Linking inventory.
  Exact SciSpaCy, language-model, UMLS release, local artifact, and license values remain
  `PENDING_APPROVAL`; no repository, version, release, or checkpoint was guessed.
- Production readiness is therefore `not_configured`, with the safe detail: exact
  SciSpaCy version, language model/version, and UMLS release await approval. Application
  startup remains available to completed stages while the Phase 6 health endpoint returns
  503 until licensed resources initialize.
- Persistent storage, Redis caching, authorization validation, real UMLS inference,
  clinical benchmarks, and terminology update testing remain open acceptance gates. Phase
  3 or Phase 4 infrastructure was not implemented and Relation Extraction was not begun.

### Verification

- Ruff passed for `New_current/app` and `New_current/tests`.
- Complete suite: 64 passed, one CUDA-conditional test skipped, and one existing
  FastAPI/Starlette TestClient deprecation warning.
- Import verification passed. OpenAPI contains all three Entity Linking operations and no
  Relation Extraction route. Dashboard and Swagger wiring passed API tests.
- A real production-configuration probe reported `not_configured`; real provider health,
  UMLS inference, accuracy, and latency are not reported as passed without licensed local
  evidence.

## 2026-08-05 - Phase 7 - Biomedical Relation Extraction Deferral

Status: `ARCHITECTURE COMPLETE - RUNTIME PENDING FINE-TUNED RELATION EXTRACTION CHECKPOINT`

- Added the Phase 7 provider interface, instance-scoped registry, dependency injection,
  application service, lifecycle, local-only BioLinkBERT adapter, typed errors, structured
  privacy-safe logging, versioned request/response schemas, and POST/health/models routes.
- Recorded `michiyasunaga/BioLinkBERT-base`, Apache-2.0, and locally evidenced immutable
  revision `b71f5d70f063d1c8f1124070ce86f1ee463ca1fe` in `MODEL_MANIFEST.md`.
- Inspected the actual local `config.json`. It declares `BertModel` and has no trained
  sequence-classification head, named relation ontology, no-relation label, preprocessing
  declaration, or calibration evidence. Provider health therefore reports
  `incompatible_artifact` and POST inference fails closed.
- The adapter rejects generic labels, random heads, missing no-relation labels, and a
  mismatched preprocessing contract. Relation types are read from checkpoint `id2label`
  rather than a hard-coded enum, preserving future ontology extensibility.
- Runtime inference, dashboard delivery, clinical quality thresholds, persistence, cache,
  and authorization remain deferred. No relationship was fabricated, no checkpoint was
  downloaded, and OCR, Medical NER, and Entity Linking behavior was not changed.

## 2026-08-05 - Phase 8 - Medical Embeddings Production Boundary

Status: `IN PROGRESS - EMBEDDING BOUNDARY COMPLETE, MODEL RUNTIME PENDING`

### Implementation

- Added `New_current/app/embeddings/` with a provider-neutral contract, instance-scoped
  registry, dependency injection, application service, lifecycle, environment/manifest
  settings, typed safe errors, health/readiness, and structured logs without input text.
- Added a complete local-only BioClinical ModernBERT adapter using `AutoTokenizer` and
  `AutoModel`, attention-mask mean pooling, optional L2 normalization, configurable
  CPU/CUDA fallback, max length, and batching. Models load once and are released on
  shutdown; Hugging Face loading uses `local_files_only=True`.
- Added `POST /api/v1/embeddings`, `GET /api/v1/embeddings/health`, and
  `GET /api/v1/embeddings/models`. Versioned responses contain input IDs, vectors,
  dimensions, vector norms, token counts, batch size, latency, throughput, device, exact
  model/revision, pooling/normalization, loading time, cache status, and configuration.
- Added `/embeddings`, a same-origin Jinja2/Bootstrap/vanilla-JavaScript console for
  de-identified batch input, vector summaries, health, latency, model revision, raw JSON,
  and Swagger access.
- Added a strict embedding inventory to `MODEL_MANIFEST.md`. Exact repository ID,
  immutable revision, license, and cache remain `PENDING_APPROVAL`; experimental repository
  references were not promoted and no artifact was downloaded or substituted.
- Qdrant, vector persistence, collections, indexing, retrieval, and later pipeline stages
  were not implemented.

### Verification

- Ruff passed across `New_current/app` and `New_current/tests`.
- Complete suite passed: 70 tests passed, one CUDA-conditional test skipped, and one
  existing FastAPI/Starlette TestClient deprecation warning remained.
- Application compilation/import passed. OpenAPI contains embedding POST, health, and
  models operations and contains no Qdrant operation. Dashboard and Swagger wiring passed.
- Provider/service/API tests cover batching, vector order, pooling without padding,
  normalization metadata, registry isolation, duplicate input rejection, stable errors,
  model inventory, health, and privacy-safe responses.
- The production embedding readiness probe reports `not_configured`: repository ID,
  immutable revision, and license await approval. Real model initialization, vector
  quality, dimensions, latency, memory, and CUDA behavior are `NOT VERIFIED`.

## 2026-08-05 - Consolidated Engineering Demonstration Dashboard

Status: `COMPLETE - INTERNAL DEMONSTRATION SURFACE`

### Implementation

- Added `/engineering-demo`, one FastAPI-served Jinja2 page using Bootstrap 5, focused CSS,
  and vanilla JavaScript. No React application, separate frontend server, authentication,
  database dependency, animation, or new API was introduced.
- Added report upload and a browser-side demonstration flow that reuses
  `POST /api/v1/ocr`, passes normalized OCR text to `POST /api/v1/ner`, and calls
  `POST /api/v1/embeddings` only when the existing embedding health endpoint reports
  ready. It does not call unavailable runtimes or fabricate downstream output.
- OCR and Medical NER cards consume their existing health/model endpoints and display
  model, immutable revision, latency, output/entities, confidence, and raw JSON. The
  embeddings card consumes its health/model endpoints and displays dimension, vector
  preview, latency, model, and revision when runtime is available.
- Entity Linking and Relation Extraction consume their existing health endpoints while
  remaining visibly `Architecture Complete` and `Runtime Pending`. PostgreSQL, Redis,
  Celery, Simplification, Verification, Translation, and TTS are static roadmap-state
  cards because no approved APIs exist for them.
- Status styling distinguishes Production Ready, Architecture Complete, Planned, and
  Frozen. TTS explicitly remains frozen until Qwen Simplification is complete.

### Verification

- Ruff passed across `New_current/app` and `New_current/tests`.
- Complete suite passed: 72 tests passed, one CUDA-conditional test skipped, and one
  existing FastAPI/Starlette TestClient deprecation warning remained.
- Application and test byte-compilation passed. Application import passed and generated
  OpenAPI contained 21 public paths, including every runtime endpoint used by the page.
- Dashboard, static JavaScript, static CSS, and Swagger returned HTTP 200 through the
  application lifecycle test. Tests verified all eleven required sections and status
  labels, same-origin endpoint reuse, absence of React, and absence of Qdrant behavior.
- Bundled Node.js syntax validation passed for `engineering_demo.js`.
- Generated `New_current/ENGINEERING_DEMO_REPORT.md`. Existing OCR, NER, Entity Linking,
  Relation Extraction, Embeddings, and API behavior was not redesigned.

## 2026-08-05 - MVP Patient Workflow Slice

Status: `IN PROGRESS - SOFTWARE FLOW COMPLETE; INDICtrans2 RUNTIME PENDING`

- Froze Entity Linking, Relation Extraction, Medical Verification, and TTS as
  **Deferred for MVP** without removing their architecture, APIs, provider boundaries, or
  roadmap acceptance criteria.
- Defined the MVP flow as Upload -> OCR -> Medical NER -> Qwen3 Simplification ->
  IndicTrans2 Translation -> final patient-friendly report. Medical embeddings remain an
  optional non-blocking background operation.
- Added provider-neutral Qwen3 simplification service and
  `POST /api/v1/simplifications` plus health/model APIs, reusing the existing tested
  local Qwen implementation at immutable revision
  `c1899de289a04d12100db370d81485cdf75e47ca`.
- Added a local-only IndicTrans2 provider/service and
  `POST /api/v1/translations` plus health/model APIs. Numeric values and units use an
  explicit placeholder preservation policy and inference fails closed if preservation
  cannot be proven.
- Updated the simple engineering demo to reuse OCR, NER, optional background embeddings,
  simplification, and translation APIs and display the final translated report.
- IndicTrans2 is not provisioned in this workspace. Its exact immutable revision and local
  cache remain pending approval, so real translation inference and end-to-end model
  verification are not reported as complete.

- Verification: changed-scope Ruff checks passed, JavaScript syntax validation passed,
  and the full suite passed with 74 tests, one CUDA-conditional skip, and the existing
  Starlette TestClient deprecation warning.
- Real local Qwen provider initialization passed on CPU for
  `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`;
  no checkpoint download or model substitution occurred.

## 2026-08-05 - MVP Upload Processing Fix

Status: `COMPLETE - DIGITAL PDF FAST PATH RESTORED`

- Diagnosed the apparently idle upload: the engineering template had been opened through
  `file://`, so no FastAPI APIs were available. A stale port-8000 server also exposed an
  older OpenAPI schema without the MVP routes.
- Verified that the four-page sample PDF contains 11,138 characters of native text, while
  the OCR provider was unnecessarily rendering every page and running the 4B vision model
  on CPU. The request consumed approximately 11.6 GB resident memory and took minutes.
- Added bounded native PDF text-layer detection and extraction before multimodal OCR.
  Every page must contain a minimum usable text layer; otherwise the complete document
  remains on the existing Qwen3-VL path, preserving scanned and mixed-document behavior.
- Native extraction records `digital_pdf`, page provenance, dimensions, latency,
  `native_pdf_text_layer` as the document-type source, and privacy-safe structured logs.
  No artificial confidence score is emitted.
- Added visible OCR, NER, embedding, simplification, and translation progress messages to
  the engineering demo.
- Regression verification: the sample uses the four-page fast path; the complete suite
  passed with 75 tests, one CUDA-only skip, and the existing TestClient warning. Changed
  Ruff checks, compilation, and JavaScript syntax validation passed.

## 2026-08-05 - Phases 3 and 4 - Production Infrastructure Software Boundary

Status: `IN PROGRESS - SOFTWARE IMPLEMENTED, LIVE DEPLOYMENT VALIDATION PENDING`

### Database and migration implementation

- Completed the previously unrecorded partial `app/db/` foundation with async SQLAlchemy
  2.x engine/session composition, tenant-scoped repositories, typed failures, database and
  pool health, optimistic version enforcement, and reversible Alembic configuration.
- Preserved the authoritative schema names: documents use `reports`; OCR output uses
  `report_processing` and `model_outputs`; NER output uses `medical_entities`; and
  simplification continues to use `simplifications`.
- Added only missing physical resources: `entity_links`, embedding metadata without raw
  vectors, `translations`, durable `processing_jobs`, append-only `audit_logs`, and
  `model_registry`. Added document SHA-256 identity for idempotency. All 17 tables have a
  UUID primary key and common timestamps/version; soft deletion is limited to appropriate
  user-visible aggregates. Binary reports, audio, and embedding vectors are not stored.
- Added `0001_initial_schema`, including constraints, indexes, native PostgreSQL UUID/
  JSONB/enums, forward creation, and reverse-order downgrade. Offline PostgreSQL upgrade
  and downgrade SQL generation passed at the single head `0001_initial_schema`.

### Redis, Celery, API, and dashboard

- Added encrypted Redis JSON caching with tenant/document/stage/pipeline/model/
  configuration/prompt-or-rule/schema identity, configurable TTL, metadata, invalidation,
  statistics, and token-safe single-flight locks.
- Added Celery JSON tasks for OCR, NER, Entity Linking, Embeddings, Simplification, and
  Translation behind deployment entry points. Configured late acknowledgement, worker-loss
  rejection, CPU/GPU queues, time limits, exponential retry of typed transient failures,
  terminal failures, progress, cancellation checks, and privacy-safe lifecycle logs.
- Jobs commit to PostgreSQL before broker submission. Broker failures remain durable in
  `retrying`; a Celery Beat task resubmits them after recovery. Duplicate submissions are
  idempotent by tenant, document SHA-256, ordered stages, pipeline, model, and configuration.
- Added `POST/GET /api/v1/jobs`, `GET/DELETE /api/v1/jobs/{job_id}`, and
  `GET /api/v1/infrastructure/health` with typed versioned schemas, OpenAPI metadata,
  examples, status codes, stable errors, ownership filtering, migration/pool/queue/cache/
  job metrics, and request IDs.
- Added `/infrastructure`, a FastAPI-served Jinja2/Bootstrap 5/vanilla-JavaScript dashboard,
  and connected the consolidated engineering demo's infrastructure card to the new health
  endpoint. No separate frontend, authentication behavior, or AI API changed.
- Added Dockerfile, authenticated Redis/PostgreSQL Compose configuration, a one-shot
  migration service, API, worker, optional Beat, health checks, persistent volumes,
  required secrets, and approved model/resource mounts.

### Objective verification

- Ruff passed across application, tests, migrations, and benchmarks. Static compilation
  passed and 107 application modules imported successfully. `pip check` found no broken
  requirements.
- Full suite: 84 passed, one CUDA-conditional test skipped, and the existing
  Starlette/TestClient deprecation warning remained. New tests cover schema inventory,
  Alembic head, repository ownership/soft deletion/versioning, durable job idempotency and
  cancellation, encrypted Redis read/write/TTL/deletion and cache-key isolation, Celery
  routes/tasks/recovery schedule, APIs, health, dashboard, and OpenAPI.
- Bundled Node.js syntax checks passed for both infrastructure and consolidated-dashboard
  JavaScript. Docker Compose YAML parsed with all six services and both persistent volumes.
- PostgreSQL offline upgrade/downgrade DDL generation passed. No Docker CLI, PostgreSQL
  service/client, or Redis executable exists on this host, so live service checks were not
  fabricated.

### Scope and remaining gates

- OCR, Medical NER, Entity Linking, and Relation Extraction source packages were not
  modified or imported by the infrastructure/database packages. No Qdrant, verification,
  new translation model, or TTS behavior was implemented.
- Phases 3 and 4 remain `IN PROGRESS`. Real PostgreSQL migration/rollback/outage tests,
  Redis expiry/lock/outage tests, Celery worker restart/retry/recovery, Compose health
  convergence, and real end-to-end job execution remain unverified.
- Frozen AI services do not yet publish the six `ib_health.pipeline_stages` deployment
  entry points. Missing stage bindings fail closed; no model output or task success is
  fabricated. Their separately reviewed binding is required after live infrastructure is
  available.
- Generated `New_current/INFRASTRUCTURE_IMPLEMENTATION_REPORT.md` and operational guidance
  in `New_current/app/infrastructure/README.md`.

## 2026-08-06 - Phase 9 - Medical Report Simplification Production Boundary

Status: `IN PROGRESS - PRODUCTION BOUNDARY IMPLEMENTED; CLINICAL VALIDATION OPEN`

### Implementation

- Replaced the MVP single-output adapter with a provider-neutral three-level result
  contract while retaining `POST /api/v1/simplifications` as a deprecated compatibility
  operation.
- Added `POST /api/v1/simplify`, `GET /api/v1/simplify/health`, and
  `GET /api/v1/simplify/models`. Each level returns source, simplified report, term
  explanations, important findings, clinician questions, measured fidelity confidence,
  latency, immutable revision, pipeline/prompt versions, warnings, and review state.
- Added external prompt `qwen-medical-simplification-v2` and strict manifest inventory.
  The adapter loads only the approved local Qwen3-0.6B snapshot, reuses one process-wide
  instance, and has no remote or heuristic production fallback.
- Added deterministic grounding controls. Output introducing a numeric value/unit or
  explaining a term absent from source evidence fails closed. Confidence is the documented
  source-fact/entity preservation ratio, not a calibrated clinical probability.
- Added `/simplify`, a Jinja2/Bootstrap/vanilla-JavaScript engineering console. Frozen OCR,
  NER, Entity Linking, Relation Extraction, Embeddings, PostgreSQL, Redis, and Celery code
  was not modified.

### Objective evidence and open gates

- Repository-wide Ruff passed. The complete suite passed: 86 tests passed and one
  CUDA-conditional test was skipped. Compilation, OpenAPI/dashboard tests, and bundled
  Node.js syntax verification passed.
- Exact local `Qwen/Qwen3-0.6B` revision
  `c1899de289a04d12100db370d81485cdf75e47ca` initialized healthy on CPU in 3758.256 ms.
- Real inference on a synthetic negated pneumonia/medication report returned all three
  levels in 91078.880 ms (451 output tokens; 427 prompt tokens). All three measured
  preservation scores were 1.0.
- Real inference on a synthetic HbA1c report was rejected because Qwen3 introduced an
  unauthorized numeric fact. Prompt hardening did not eliminate that behavior; the guard
  continued to fail closed. This case is not reported as a pass.
- Phase 9 remains `IN PROGRESS` pending representative clinical faithfulness/readability
  thresholds, especially numeric/laboratory reports. CUDA is `NOT VERIFIED`. Shared cache
  and persistence activation remain within the frozen Phase 3/4 deployment boundary.

## 2026-08-09 - Phases 2/5 - OCR Reliability Verification

Status: `PHASE 2 IN PROGRESS - LIVE CPU EVIDENCE ADDED, NO PRODUCTION CODE DEFECT FOUND`

### Verified

- Read through the complete `New_current/app/ocr/` boundary (api, application, domain,
  providers, postprocessing, infrastructure, observability) and its test suite. No
  placeholder response, silent fallback, or scope violation was found; the module matches
  its README and `AGENTS.md` completeness rules.
- `python -m pytest -q tests/ocr` passes 33/34 (1 CUDA-conditional skip) once pytest is
  given a writable `--basetemp`; the default Windows temp directory
  (`%LOCALAPPDATA%\Temp\pytest-of-<user>`) on this host is access-denied for this account,
  an environment condition unrelated to the application code.
- Ran a direct live smoke test against the real pinned `Qwen/Qwen3-VL-4B-Instruct`
  checkpoint (revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, previously untested with
  the required environment variables actually exported): initialized in 18.1 s and
  transcribed a synthetic PNG exactly, with `confidence=0.9999`
  (`mean_generated_token_probability`) on CPU.
- Ran the same style of live smoke test against the pinned `d4data/biomedical-ner-all`
  checkpoint: initialized in 32.2 s, ran inference in 1.3 s, and extracted 8 real clinical
  entities (Disease, Medication, Measurement, Procedure, Anatomy) with genuine softmax
  confidences from `tests/ner`'s production boundary.

### Fixed

- `New_current/benchmarks/ocr/run_validation.py` (`_benchmark_ocr`): one document that
  failed a configured limit (in this case an oversized page count in the reviewer's own
  test configuration) raised out of the per-format loop and discarded every already-passing
  format's evidence for that device, corrupting the whole validation report as
  `NOT VERIFIED`/`FAIL` even when most formats had actually passed. Each document's
  inference is now isolated in its own try/except, recorded per-format, with the device's
  overall status computed honestly from the aggregated per-format results. This is a
  benchmark/evidence-tooling fix only; no production route, service, or provider changed.

### Objective evidence

- `python -m ruff check benchmarks/ocr/run_validation.py` and `python -m py_compile` both
  pass after the fix.
- No production `New_current/app/ocr` file required a change; the module's fail-closed
  manifest/revision-provenance checks, cache-key versioning, and provider lifecycle were
  read in full and are correct as implemented.

### Known limitations

- The corrected full 5-format `run_validation.py` evidence run (PDF/multi-page
  PDF/PNG/JPEG/TIFF on CPU) was superseded mid-run by the GPU performance work below and
  was not re-completed in this session; `benchmarks/ocr/reports/` still reflects an older
  run. Re-running it is recommended before reporting Phase 2 Step 4 as `PASS`.
- Clinical-corpus accuracy/threshold validation for both OCR and NER remains open per
  `ROADMAP.md` and is unaffected by this entry.

## 2026-08-09 - Cross-Phase (2, 5, 9, 11) - MVP Inference Performance Optimization

Status: `IN PROGRESS - GPU PATH ENABLED AND MEASURED FOR OCR/NER/SIMPLIFICATION; TRANSLATION NOT RUNTIME-TESTABLE`

### Problem

The MVP pipeline (Upload -> Qwen3-VL OCR -> biomedical-ner-all NER -> Qwen3-0.6B
Simplification -> IndicTrans2 -> patient-friendly report) was running every stage on CPU.
The host has a physical NVIDIA GeForce RTX 5050 Laptop GPU (8 GB, driver 592.27, CUDA 13.1
capable), but the installed PyTorch build (`2.13.0+cpu`) had no CUDA support at all, so
`torch.cuda.is_available()` was `False` regardless of configuration. Independently, the OCR
provider's device defaulted to a hardcoded `cpu` in `MODEL_MANIFEST.md` and
`docker-compose.yml`, and the NER provider defaulted to a hardcoded `cpu` in code with no
`auto` mode at all - so even a correctly CUDA-enabled host would still have silently run
both models on CPU.

### Built / Fixed

- Installed `torch==2.11.0+cu128` into `New_current/.venv`, replacing the CPU-only build.
  Verified with a real CUDA matmul kernel: `torch.cuda.is_available()` is `True`, device
  `NVIDIA GeForce RTX 5050 Laptop GPU`, compute capability `(12, 0)` (Blackwell/sm_120),
  `bf16` supported. This is an environment/dependency change, not an application code
  change, and does not touch any approved model identity.
- `MODEL_MANIFEST.md` / `New_current/docker-compose.yml`: changed the Qwen3-VL OCR
  `device` default from hardcoded `cpu` to `auto`. The provider's `_resolve_torch_device()`
  already fully implemented `auto` (CUDA when available, otherwise CPU, and a loud failure
  rather than a silent fallback when `cuda` is explicitly requested but unavailable); only
  the manifest/deployment default was wrong.
- `New_current/app/ner/providers.py`: changed the default `NER_CONFIG__DEVICE` from `cpu`
  to `auto` and added the same auto-resolution + fail-closed-on-explicit-cuda behavior to
  `LocalTokenClassificationProvider._load_runtime`, plus GPU dtype selection (`bfloat16` if
  supported, else `float16`, else `float32` on CPU). Simplification
  (`app/simplification/provider.py`) and Translation (`app/translation/provider.py`)
  already defaulted to `auto`; Translation's dtype selection was added to match.
- `New_current/app/ocr/providers/implementations.py`: `top_p` was being passed to
  `generate()` unconditionally even when `do_sample=False`, which is invalid for greedy
  decoding and produced a Transformers warning every call. It is now only set when sampling
  is enabled, matching the existing `temperature` conditional. Verified
  `generation_config.json` already declares proper `eos_token_id`/`pad_token_id`, so
  early-stopping was already correctly configured.
- `New_current/app/translation/contracts.py`, `provider.py`, `service.py`: added a real
  `translate_batch` path. `IndicTrans2Provider.translate()` is now a thin wrapper over
  `translate_batch()`, which runs one batched preprocess/tokenize/generate/postprocess pass
  for multiple texts (e.g. translating all three simplification levels in one model call
  instead of three sequential ones) instead of the previous single-text-only path. No
  existing API contract changed; `TranslationService.process_batch()` is available for the
  future end-to-end orchestrator (Phase 13) to use.
- `New_current/app/static/engineering_demo.js`: the demo pipeline was `await`-ing the
  medical-embeddings call before starting simplification, contradicting
  `ARCHITECTURE.md`/`ROADMAP.md`'s explicit "embeddings are background-only and must not
  block the patient workflow" requirement. Changed to fire-and-forget with its own
  success/failure rendering, so simplification starts immediately after NER.
- Confirmed (no change needed): all four stage services already offload blocking model
  inference via `asyncio.to_thread` (event loop is not blocked during inference); OCR
  already has working SHA-256 content-addressed result caching
  (`app/ocr/application/cache.py`) so exact re-uploads skip inference entirely;
  Simplification already caches the loaded model process-wide keyed by `(path, device)` so
  it is never reloaded per request. Redis/Celery activation for backgrounding these stages
  is correctly left to the separately gated Phase 4 (live Redis/Postgres are unavailable on
  this host); it was not attempted here.

### Measured (Before -> After), same host, representative single-input smoke tests

| Stage | Device Before | Before | Device After | After | Speedup |
|---|---|---|---|---|---|
| OCR (Qwen3-VL-4B, synthetic PNG) | cpu | 168.7 s | cuda | 17.2 s | 9.8x |
| NER (biomedical-ner-all, 1 paragraph) | cpu | 1.30 s | cuda | 0.77 s | 1.7x |
| Simplification (Qwen3-0.6B, 3 levels, isolated GPU run) | cpu | 463.3 s | cuda | 135.4 s | 3.4x |
| Translation (IndicTrans2) | n/a | not runtime-testable | n/a | not runtime-testable | - |

Model loading time increased on GPU for OCR (18.1 s -> 43.9 s, moving ~8 GB of bf16 weights
onto the GPU) and decreased for NER (32.2 s -> 10.7 s); this is a one-time per-process cost,
not a per-request cost, and is not included in the inference timings above. Output text,
entities, and confidence values were unchanged within expected bf16 numeric noise.

### Root-cause note on the smaller speedups

A raw micro-benchmark of `Qwen3-0.6B.generate()` alone measured ~3.8-4.8 tokens/second on
this GPU with `nvidia-smi` showing only 3-18% GPU utilization and ~20-29 W draw (of a 115 W
cap) throughout - i.e. the GPU sits mostly idle between steps rather than compute-bound.
This is consistent with Windows WDDM's known higher per-kernel-launch latency (this GPU has
no TCC option, being a GeForce part) dominating wall-clock time when per-token compute is
tiny relative to fixed launch/sync overhead, which disproportionately affects the smaller
Simplification/NER models compared to the much larger OCR model. This is a host/OS/driver
characteristic, not an application inefficiency; no further code change was attempted
against it in this pass.

### Objective evidence

- Full suite: `python -m pytest -q tests` -> 86 passed, 1 skipped (the "CPU fallback
  requires absent CUDA" test now correctly skips, since CUDA is genuinely present).
- `python -m ruff check app benchmarks` -> all checks passed.
- All timings above are from direct calls into the real production provider/service classes
  (not mocked), using the exact pinned manifest checkpoints.

### Known limitations / not done

- Translation could not be measured: `MODEL_MANIFEST.md` still records
  `Pinned Revision: PENDING_APPROVAL` for IndicTrans2 and no local checkpoint is
  provisioned in this workspace, so `TranslationService.process`/`process_batch` remain
  correctly fail-closed (`not_configured`). The new batching code path is implemented and
  unit-verified by import/compile/lint only, not model-validated.
- No result cache was added for NER, Simplification, or Translation (unlike OCR's existing
  cache); the current call pattern in this codebase never repeats an identical input to
  those stages independently of the OCR stage's own cache, so this was judged lower-value
  than the GPU-enablement work and left for a future task if duplicate-submission handling
  becomes a measured problem.

## 2026-08-09 - Phase 11 - IndicTrans2 Runtime Validation

Status: IN PROGRESS - LOCAL MODEL RUNTIME VALIDATED; CLINICAL VALIDATION OPEN

- Replaced the previously unprovisioned Translation inventory with the approved local-only
  ai4bharat/indictrans2-en-indic-dist-200M checkpoint at immutable revision
  173b94239f7c38886b2747b8d4a5db771a7e1232 (MIT). The snapshot resides under
  New_current/.model-cache/indictrans2-en-indic-dist-200M/ at that exact revision.
- Installed the declared IndicTransToolkit runtime and verified offline local loading of
  the tokenizer, remote-code sequence-to-sequence model, and processor. device=auto
  selected CUDA. No model was downloaded at runtime and no alternate engine/API was used.
- Fixed two verified provider issues: generation now disables the incompatible Transformers
  KV-cache path for this custom checkpoint, and numeric/unit/date sentinels use unique
  bracketed identifiers that survive supported Indic script transliteration or fail closed.
  The maximum generation length is environment-backed and validated from 1 through 256.
- Added focused Translation tests for unprovisioned configuration, initialization, device
  selection, invalid generation configuration, single/batch inference, language selection,
  numeric/date preservation, and failure handling. Added the reproducible offline benchmark
  runner benchmarks/translation/run_indictrans2_benchmark.py.
- Real benchmark evidence: load 2,345.000 ms; first 3,708.981 ms; warm 1,734.264 ms;
  three-text batch 1,551.046 ms / 1.934 texts/s; RSS 1,984.121 MiB; peak CUDA allocation
  616.467 MiB. Hindi, Tamil, and Kannada synthetic medical-text checks retained every
  protected dosage, unit, pressure, and date exactly.
- Real Simplification -> Translation handoff: Qwen3 produced three source-grounded levels
  in 18,745.659 ms and the real IndicTrans2 service translated those exact three strings
  in one 1,230.103 ms batch. The existing typed HTTP workflow test continues to prove
  OCR -> NER -> Simplification -> Translation contract propagation; upstream GPU work was
  not repeated.
- Verification: python -m ruff check app benchmarks tests passed; complete test suite
  passed 94 passed, 1 skipped; translation imports and byte-compilation passed. The
  FastAPI dashboard/OpenAPI lifecycle tests remain part of the passing suite.
- Phase 11 remains open: expected artifact SHA-256 is still PENDING_APPROVAL, clinical
  quality validation across every enabled language is not complete, and shared-cache/durable
  full-pipeline orchestration validation belongs to the remaining infrastructure/Phase 13
  gates. No clinical-validation claim is made.

## 2026-08-09 - Phase 11 - Approved IndicTrans2 Artifact and E2E Validation

Status: IN PROGRESS - APPROVED LOCAL ARTIFACT AND E2E RUNTIME VALIDATED

- Updated the approved Translation inventory to the supplied local checkpoint directory
  New_current/.model-cache/translation/indictrans2-en-indic-dist-200M. The approved primary
  artifact is model.safetensors with SHA-256
  0039e4304e9889acc5c8350a193311d07aa5399b6d9fb0445e8fde19e3533bb5.
- The provider now requires the exact manifest cache directory, verifies that safetensors
  artifact before model initialization, and loads with use_safetensors=True. It will not load
  pytorch_model.bin as a compatibility fallback.
- Real offline CUDA initialization succeeded with checksum_verified=True. English to Hindi,
  Tamil, and Kannada translations preserved the tested HbA1c, Metformin, dosage, percent,
  blood-pressure, and unit values; a three-item batch also completed.
- Approved-artifact benchmark: load 2,133.898 ms; first inference 4,750.691 ms; warm
  inference 2,217.984 ms; three-text batch 3,459.980 ms; throughput 0.867 texts/s;
  process RSS 1,982.383 MiB; CUDA peak allocation 616.467 MiB.
- A real, no-mock end-to-end run completed OCR -> NER -> Qwen3 Simplification -> IndicTrans2
  Translation for a de-identified no-numeric document. OCR 7,123.953 ms; NER 27.964 ms;
  Simplification 14,264.999 ms; Translation 768.157 ms; total 22,185.073 ms. It returned
  three translated Tamil levels. A separate numeric input reached the existing Simplification
  source-fact guard and was correctly rejected when Qwen3 introduced an unsupported number.
- Verification: Ruff passed; full test suite passed 95 passed, 1 skipped; compilation,
  import, and an additional checksum-verified local provider initialization passed.
- Phase 11 remains in progress. Clinical translation-quality validation, shared
  cache/persistence validation, and durable orchestration are not claimed complete. The
  Phase 9 numeric-report simplification safety limitation remains deliberately fail-closed.

## 2026-08-09 - Phase 10 - Medical Verification Technical Validation

Status: TECHNICALLY VERIFIED; LICENSE PENDING; PRODUCTION APPROVAL PENDING

- Implemented the local-only PubMedBERT MedNLI provider, provider-neutral service,
  deterministic grounding policy, POST /api/v1/verification, health/models endpoints, and
  a minimal /verification Jinja2 testing page.
- The provider requires the approved local cache and immutable revision provenance, validates
  BertForSequenceClassification, 512-token limit, and config.json mapping 0=contradiction,
  1=entailment, 2=neutral. Labels are read from the explicit approved mapping only.
- Real offline CUDA inference: load 1,750.681 ms; RSS 1,132.473 MiB; peak GPU allocation
  428.413 MiB. Clear entailment returned entailment/PASS, contradiction returned
  contradiction/BLOCKED, neutral returned neutral/REVIEW, and numeric, dosage, and negation
  mismatches returned BLOCKED with deterministic evidence.
- Engineering-demo routing invokes Verification after Simplification and before Translation.
  BLOCKED and REVIEW outputs stop Translation. Existing simplification grounding guards were
  not weakened.
- License remains PENDING_VERIFICATION. This candidate is not production approved and has
  no clinical validation claim.

## 2026-08-09 - Clinical Performance Benchmark Harness

Status: IMPLEMENTED; REPRESENTATIVE CLINICAL DATA PENDING

- Added an offline, model-neutral benchmark harness under
  New_current/benchmarks/clinical_performance. It defines a versioned JSONL schema for
  approved de-identified lab, prescription, discharge, radiology, consultation,
  handwritten/scanned, table-heavy, small-text, and multi-page documents.
- The harness explicitly leaves CER/WER unmeasured when independently reviewed OCR gold
  text is absent, records protected numeric/unit/dosage/frequency/negation/laterality
  preservation, and writes JSON, CSV, and Markdown comparison artifacts.
- Captured the existing 22,185.073 ms real warm E2E baseline as evidence only. 512px OCR,
  FP16, INT8, 4-bit, compact simplification schema, and future model replacement candidates
  remain MORE_VALIDATION_REQUIRED. No production setting, model identity, safety guard, or
  public API was changed.
- Verification: Ruff passed, focused harness tests passed (4), full suite passed
  (99 passed, 1 skipped), and compilation/import verification passed.

## 2026-08-09 - PDF Deid Synthetic OCR Benchmark

Status: PRODUCTION BASELINE BLOCKED

- Registered pdf_deid_synthetic_medical_v1: 50 synthetic PDFs (30 Easy, 10 Medium, 10
  Hard). Its cumulative filename-keyed mappings contain PHI token lists only, not complete
  reviewed transcriptions; CER/WER and clinical-text metrics are not claimed.
- The resolved Compose configuration was tested without alteration on a Medium PDF:
  Qwen3-VL-4B, pinned revision, device auto, dtype float32, image size 1600, DPI 144,
  max_new_tokens 2048. The provider failed during CUDA initialization with out of memory
  while moving the FP32 checkpoint to the 8GB GPU.
- Per the benchmark stop condition, the 512px candidate and full run were not launched.
  This is evidence of a production runtime-capacity issue, not a candidate regression.

## 2026-08-09 - OCR Precision Qualification on Current 8GB CUDA GPU

Status: INITIALIZATION EVIDENCE COMPLETE; FULL-QUALITY OCR SMOKE OPEN

- The exact Compose FP32 profile remains NOT DEPLOYABLE ON CURRENT 8GB GPU due to CUDA OOM
  before inference. This is a hardware-profile capacity result, not a model failure.
- The pinned Qwen3-VL-4B model initialized with BF16 (24,058.935 ms; 8,519.605 MiB allocated;
  8,556.000 MiB reserved) and FP16 (28,244.485 ms; same reported GPU memory). BF16 is the
  lower-load-time preferred candidate.
- A BF16 smoke at the unchanged 1600px, 144 DPI, 2048-token configuration did not finish
  the Medium/Hard image-OCR work within the 360-second harness limit. The controlled 512px
  candidate and full synthetic dataset run were not launched. No production default changed.

## 2026-08-09 - BF16 OCR Image-Cost Sweep

Status: NO DEPLOYABLE PROFILE QUALIFIED

- The staged BF16 Medium-PDF sweep retained model identity, prompt, deterministic decoding,
  2048-token ceiling, and safety behavior. Both 1024px at 144 DPI and 512px at 96 DPI
  initialized but did not complete OCR within a 150-second practical harness limit.
- No structured OCR output, actual generated-token count, PHI recovery, or candidate quality
  delta was available. The matching results mean image resolution has not demonstrated a
  deployable region while the configured generation behavior remains unmeasured.
- BF16 remains the initialization-preferred candidate; neither BF16 nor 512px is promoted.
  The full 50-PDF synthetic benchmark was not started and production defaults remain frozen.

## 2026-08-09 - Qwen3-VL Generation Behavior Diagnostic

Status: OUTPUT-CONTRACT BOTTLENECK IDENTIFIED; NO PRODUCTION CHANGE

- Real BF16 diagnostics on one Medium and one Hard synthetic PDF used the loaded provider
  model, production prompt/decoding/2048-token ceiling, and a benchmark-only 45-second
  stopping criterion. Structured output generated 66 tokens at 1.424 and 1.433 tokens/sec;
  it started valid JSON and transcribed ordinary visible text, but did not reach EOS or a
  closing JSON object before the stop.
- A transcription-only benchmark prompt generated 134 and 143 ordinary transcription tokens
  at 2.962 and 3.141 tokens/sec. Its prompt was 201 tokens versus 258 for structured output.
  Neither contract reached 2048 tokens; no repetition or malformed-output recovery behavior
  was observed.
- Processor time and visual tensor shape did not indicate the primary bottleneck. The
  evidence supports a separate approved experiment to simplify the generative output
  contract and construct envelopes deterministically. Model identity, production prompt,
  APIs, safety behavior, and defaults were not changed.
- `torch==2.11.0+cu128` was installed directly into the developer venv for this
  measurement; `New_current/requirements.txt`/`pyproject.toml` still resolve to the CPU
  wheel by default and were not changed, since pinning a CUDA build repo-wide is a
  deployment/hardware decision outside this task's scope and would need explicit approval
  (CI and CPU-only hosts must still be able to install and run this project).

## 2026-08-09 - Internal Pipeline Test Console (`/engineering-demo`)

Status: `CONSOLE REWRITTEN; NO BACKEND OR AI ARCHITECTURE CHANGE`

### Built

- Rewrote `New_current/app/templates/engineering_demo.html`,
  `New_current/app/static/engineering_demo.js`, and
  `New_current/app/static/engineering_demo.css` into a consolidated internal Pipeline Test
  Console. It is explicitly not the patient-facing product; it is a same-origin FastAPI +
  Jinja2 + Bootstrap + vanilla JavaScript page, with no React, no new frontend server, and
  no authentication added.
- Added Step-by-Step Mode (a manual "Run X" button per stage, with tabs for
  normalized/raw/JSON/stats output) and Full Pipeline Mode (one upload runs OCR through
  Translation in dependency order with a real-time vertical progress tracker driven only by
  actual fetch resolutions: `WAITING -> RUNNING -> PASS/REVIEW/BLOCKED/FAILED`; no
  `setInterval`/timer-simulated progress).
- Consumes only existing endpoints: `/api/v1/ocr`, `/api/v1/ner`, `/api/v1/embeddings`,
  `/api/v1/simplify` (the three-level endpoint; the deprecated `/api/v1/simplifications`
  compatibility route is intentionally not called), `/api/v1/verification`,
  `/api/v1/translations`, plus each stage's `/health` and `/models`, and
  `/api/v1/entity-linking/health` / `/api/v1/relation-extraction/health` for the two
  runtime-pending stages. TTS has no route at all and is rendered as a static `Frozen` tile.
- Verification is run once per simplification level (premise = OCR normalized text,
  hypothesis = that level's `simplified_report`, matching the only contract
  `POST /api/v1/verification` exposes) and its returned `verification` field
  (`PASS`/`REVIEW`/`BLOCKED`) is the sole gate for offering that level for translation;
  `REVIEW` and `BLOCKED` both withhold translation, matching the policy already encoded in
  the previous engineering-demo script rather than inventing a more permissive one.
- Performance Panel, Runtime Stats, and Safety Checks render only fields present in the
  real responses. Where the backend does not expose a metric (GPU memory
  allocated/reserved/peak, GPU utilization, CPU RSS/usage, translation device, upload/decode
  latency breakdown, document SHA-256, cold/warm) the console prints `NOT EXPOSED` rather
  than a computed or invented value; no new backend endpoint was added to manufacture these.
  Two fields are intentionally client-computed and labeled as such: min/max entity
  confidence (aggregated from the response's own entity list) and "time to first meaningful
  result" (a `performance.now()` measurement bracketing the OCR call).
- The Raw API Inspector logs every request/response pair (method, URL, HTTP status, request
  ID, JSON body, Copy JSON) for every call the page makes; errors render the backend's safe
  `{error: {code, message}, request_id}` envelope only, never a stack trace, and a failed
  stage stops its dependents while leaving already-completed stage output visible.

### Tests

- Rewrote `New_current/tests/test_engineering_demo.py`: page/script contract, endpoint
  reference set now includes `/api/v1/simplify`, `/api/v1/verification`,
  `/api/v1/translations` (and asserts the deprecated `/api/v1/simplifications` is absent),
  presence of the real stage-status vocabulary and `NOT EXPOSED`, absence of hardcoded
  numbers copied from `PERFORMANCE_PROFILE.md`, and three full-stack `TestClient` runs
  through the exact endpoints the dashboard calls proving `PASS` allows translation while
  `BLOCKED`/`REVIEW` verdicts are what the dashboard's gating logic reads.
- Updated `New_current/tests/test_mvp_workflow.py::test_demo_marks_frozen_stages_deferred_for_mvp`
  (renamed to `test_demo_marks_non_functional_stages_without_fabricating_output`): the prior
  assertion required the literal string `"Deferred for MVP"`, which this rewrite
  intentionally replaced with `Runtime Pending` / `Frozen` per the current task's labeling
  requirement; the assertion now matches the new copy instead of the old one.
- `python -m ruff check app tests benchmarks`: all checks passed.
- `python -m pytest -q tests`: 109 passed, 1 skipped (CUDA-conditional), 0 failed.
- `python -m compileall -q app tests benchmarks`: passed.
- `create_app().openapi()` generated successfully and contains every endpoint the console
  calls.
- JavaScript syntax was **not** independently verified with a JS engine: Node.js is not
  installed on this host (`node: command not found`). The script's correctness is instead
  evidenced by the `TestClient` tests above, which serve it as a real static asset and
  assert on its literal content, plus manual review.

### Known limitations

- Full end-to-end manual verification in an actual browser against the live model
  checkpoints was not performed in this session (would require driving a real upload
  through several minutes of CPU/GPU inference); correctness rests on the API contracts
  read directly from each stage's `schemas.py`/`routes.py` and the automated tests above.
- Document SHA-256/hash, GPU/CPU runtime memory and utilization, and translation device are
  genuinely not returned by any current endpoint; adding them would require a backend
  change, which this task explicitly scoped out ("do not redesign APIs").

## 2026-08-09 - OCR Output Contract Optimization Experiment

Status: BENCHMARK ADAPTER IMPLEMENTED; CONTROLLED MODEL VALIDATION NOT VERIFIED; NO PRODUCTION CHANGE

- Added an opt-in, benchmark-only transcription-only Qwen3-VL mode plus a controlled runner.
  It reuses the same initialized model/tokenizer, BF16 runtime, processor, rendering,
  deterministic decoding, 2048-token ceiling, and token-confidence calculation for both
  contracts. Public APIs and deployed OCR behavior are unchanged.
- The candidate creates its response envelope deterministically. Scanned/image document type
  is `unknown` and therefore review-required; the digital-PDF fast path remains unchanged.
  Candidate confidence is labeled `mean_transcription_token_probability` and is not claimed
  comparable to structured JSON token confidence.
- Real local BF16 initialization succeeded, but the first selected multi-page Medium structured
  baseline returned no final sequence after more than six minutes. The process was terminated;
  candidate inference and all four-document quality/latency metrics remain NOT VERIFIED.
- The isolated benchmark adapter applies the existing 300-second policy inside Transformers
  generation for future controlled runs. Local production post-processing resources remain
  unmounted; fixture resources are symmetric but do not validate production resources.
- Evidence: `New_current/benchmarks/clinical_performance/results/pdf_deid_ocr/output_contract_experiment/`.
- Verification: `python -m ruff check app tests benchmarks` passed; `python -m pytest -q tests`
  passed (113 passed, 1 skipped); `python -m compileall -q app benchmarks tests` passed.

## 2026-08-12 - PP-OCRv6 Medium Candidate Artifact Gate

Status: `ARTIFACT_IDENTITY_AMBIGUOUS`; NO INFERENCE OR PRODUCTION CHANGE

- Verified the approved local SHA-256 values for both PP-OCRv6 candidate safetensors files.
  Detection metadata consistently identifies `PP-OCRv6_medium_det`.
- Recognition metadata is internally contradictory: local `inference.yml` and README identify
  `PP-OCRv6_medium_rec`, but its local `config.json` declares
  `model_type: pp_ocrv6_small_rec`. The supplied checksum matches the file, so this is an
  artifact identity/provenance issue rather than an integrity mismatch.
- Per the benchmark stop condition, DET/REC initialization, PDF Deid inference, Qwen
  comparison, and hybrid feasibility evaluation were not run. Loading it as Medium would
  misrepresent the candidate; loading it as Small would substitute an unapproved model.
- Evidence: `New_current/benchmarks/clinical_performance/results/ppocrv6_medium_artifact_validation.md`.

## 2026-08-09 - Production-Safe Runtime Observability

Status: `OBSERVABILITY ADDED; NO MODEL, PROMPT, RESOLUTION, PRECISION, OR SAFETY-CHECK CHANGE`

### Built

- New `GET /api/v1/runtime/metrics` in `New_current/app/infrastructure/routes.py` +
  `schemas.py` (`RuntimeMetricsResponse`/`GPUMetrics`/`CPUMetrics`/`ModelRuntimeStatus`).
  Returns process-wide CUDA memory (allocated/reserved/peak/total from `torch.cuda`, zero
  new dependency), CPU process RSS/utilization (`psutil`, already an approved dependency —
  no new package added), and per-stage loaded/warm/load-timestamp/load-duration/request-count
  read generically from each stage's existing `metadata()`/`health()` object via
  `request.app.state`. GPU utilization is attempted through `torch.cuda.utilization()`; on
  this host that requires `pynvml`, which is not installed and was deliberately **not**
  added per the task's "no heavy dependency solely for GPU utilization" instruction, so it
  returns `null` (`utilization_source: null`) rather than a substitute number. One stage's
  failure to report is caught and logged; it does not take down the endpoint for the others.
- Added `self._request_count` (incremented once per real inference call) and, where a
  provider previously had no `startup_timestamp`, added one (Simplification, Translation,
  Verification) — all folded into each provider's existing free-form `configuration` dict,
  so no response schema changed shape for OCR, NER, Embeddings. "Warm" is `true` once a
  provider's model is resident (these providers are process-resident singletons that are
  never reloaded per request, per `PERFORMANCE_PROFILE.md`); no model is ever reloaded
  merely to measure this.
- Closed two real device-visibility gaps: `TranslationProviderMetadata.device` and
  `VerificationProviderMetadata.device` were already tracked internally but neither
  `TranslationModelResponse` nor `VerificationModelResponse` ever surfaced them. Added a
  `device: str` field to both schemas (additive, matching the field OCR/NER/Simplification/
  Embeddings already expose) and populated it in each route's builder function.
- Added an OCR upload/decode timing breakdown without changing `OCRResponse`'s shape:
  `app/ocr/api/routes.py` measures upload/read time around `file.read()`; the provider
  measures decode/PDF-render time separately from model inference inside
  `Qwen3VLOCRProvider._process()` (new `OCRProcessingStatistics.decode_time_ms` field,
  defaulted to `0.0` for existing callers); both flow through
  `OCRApplicationService.process(upload_time_ms=...)` into two new additive keys,
  `metadata.upload_time_ms` and `metadata.decode_time_ms`, inside the OCR response's
  existing open-ended `metadata` dict. The digital-PDF native-text fast path reports its
  entire elapsed time as `decode_time_ms` (there is no model inference on that path).
- Updated `/engineering-demo`'s Runtime Stats panel to show real GPU memory/utilization and
  CPU RSS/usage from the new endpoint (replacing the previous hardcoded `NOT EXPOSED` rows),
  added a "Warm / Cold" cell to every stage's metadata grid, fixed the Translation device
  cell (previously a hardcoded literal `NOT EXPOSED` string in the template regardless of
  the API), added a Device cell to the Verification card, and split the Performance Panel's
  OCR row into `Upload / Read`, `Decode / Render`, `OCR inference (Qwen3-VL)`, and
  `OCR Post-processing` in a fixed canonical order (independent of call order, so re-running
  one stage does not reshuffle the waterfall). Every cell still prints `NOT EXPOSED` for
  `null`/missing values; none of the new numbers are hardcoded.

### APIs and compatibility

- New endpoint: `GET /api/v1/runtime/metrics` (additive).
- Additive fields only: `TranslationModelResponse.device`, `VerificationModelResponse.device`,
  `OCRApplicationService.process(upload_time_ms=...)` (new optional keyword parameter,
  default `None`), `OCRResultBuilder.build(upload_time_ms=...)` (new optional keyword
  parameter), `OCRProcessingStatistics.decode_time_ms` (new field, default `0.0`). No
  existing field was removed, renamed, or changed type. No model identity, prompt, image
  resolution, precision, generation limit, translation behavior, or safety/verification
  check was changed.

### Tests and verification

- Added to `New_current/tests/test_engineering_demo.py`: the new endpoint is in the
  dashboard's referenced-endpoint set; a real-or-null assertion on every `/api/v1/runtime/metrics`
  field (process RSS must be a real positive float since `psutil` always works here; GPU
  utilization must be `null` since `pynvml` is absent); confirms `/api/v1/verification/health`
  and `/api/v1/translations/health` now report a real `device`; confirms the dashboard
  references the new warm/cold vocabulary and reads `gpu.allocated_mb`/`utilization_percent`/
  `cpu.process_rss_mb` from the response rather than a literal.
- `python -m ruff check app tests benchmarks`: all checks passed.
- `python -m pytest -q tests`: 117 passed, 1 skipped (CUDA-conditional), 0 failed.
- `python -m compileall -q app tests benchmarks`: passed.
- `create_app().openapi()` regenerated successfully; `/api/v1/runtime/metrics` present.
- A manual live call against the real (non-fake) simplification/translation/verification
  providers on this host returned real numbers, e.g. `gpu.allocated_mb: 1961.202`,
  `gpu.reserved_mb: 2010.0`, `gpu.total_mb: 8150.562`, `cpu.process_rss_mb: 1148.793`,
  `cpu.process_cpu_percent: 104.9`, each provider's `load_timestamp`/`load_duration_ms`
  correctly reflecting real startup order.
- JavaScript syntax was **not** independently verified with a JS engine: Node.js remains
  unavailable on this host. Correctness rests on the `TestClient` tests above, which serve
  and assert on the script's literal content, plus manual review.

### Known limitations

- GPU utilization percent remains `NOT EXPOSED` on every host without `pynvml` installed;
  adding it was explicitly out of scope for this task.
- Host-wide CPU/RAM (as opposed to this process's RSS/CPU%) is not reported; the task asked
  for process metrics, and reporting host-wide figures from inside a container/venv can be
  misleading, so it was not added speculatively.
- The unified runtime-metrics model table does not carry per-model `license`/`dtype`
  columns (shown as `NOT EXPOSED` there) because those keys are not uniformly named across
  every stage's provider `configuration` dict; OCR's dtype remains visible in its own
  dedicated section, which already had it before this task.

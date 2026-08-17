# IB Health Engineering Context

This file is the standing engineering prompt for every contributor and coding assistant
working in this repository. Before implementation, read `ARCHITECTURE.md`, `ROADMAP.md`,
and `IMPLEMENTATION_LOG.md`. Do not restate the entire project context in task prompts.

## Product

IB Health Medical Term Simplifier converts clinical documents into patient-friendly
explanations without losing medical meaning. Supported product inputs include laboratory
reports, discharge summaries, prescriptions, radiology reports, and consultation notes.
The intended output may include a simplified report, medical explanations, translation,
and speech.

This is production healthcare software, not a prototype, hackathon project, proof of
concept, or research notebook. Optimize for correctness, traceability, privacy,
maintainability, and safe failure. Never present generated text as diagnosis or treatment
advice.

## Source of truth and scope

- `ARCHITECTURE.md` is the source of truth for the pipeline, target model stack, existing
  database schema, service boundaries, API conventions, and engineering standards.
- `ROADMAP.md` is the engineering contract for implementation order, deliverables, phase
  status, and acceptance criteria. Work only within the requested active phase.
- `IMPLEMENTATION_LOG.md` is the durable record of completed work, migrations, APIs,
  models, measurements, verification, and remaining limitations.
- `DB.pdf` is the original schema diagram. Do not redesign that schema. Add a table only
  when the requested module cannot be implemented with the existing schema, and document
  the reason in an Alembic migration.
- `New_current/` is the current production-oriented implementation. Other top-level
  directories contain experiments, evaluation utilities, training data, or earlier work
  unless a task explicitly puts them in scope.
- Implement only the requested module. Do not implement later pipeline stages speculatively.
- Preserve existing API contracts unless the task explicitly authorizes a breaking change.
- When documentation and executable code disagree, report the mismatch and update the
  appropriate source rather than silently assuming the planned state is implemented.

## Phase protection

- Never rewrite a completed module or phase.
- Modify a completed phase only to fix a verified bug, improve measured performance, or
  add an interface required by the active phase. Keep the change as small as possible and
  add regression tests.
- Do not perform broad cleanup, dependency replacement, package reorganization, or
  large-scale refactoring without explicit instruction.
- Preserve behavior outside the requested scope. If an unavoidable cross-phase change is
  required, explain it before implementation and record it in `IMPLEMENTATION_LOG.md`.
- A phase is complete only when every deliverable and acceptance criterion in
  `ROADMAP.md` is satisfied and the implementation log has been updated. Existing partial
  or experimental code does not make a roadmap phase complete.

## Required architecture

Each AI stage is independently replaceable and has a clear API, service, provider
interface, schemas, configuration, and tests. API routes orchestrate dependencies; they
must not instantiate or call model libraries directly. Use dependency injection and keep
provider-specific details behind interfaces such as `BaseOCRProvider` or an equivalent
typed protocol.

New public endpoints use `/api/v1/`. Expose only HTTP methods that make semantic sense.
Every endpoint must include an OpenAPI summary, description, typed request and response
models, examples, expected status codes, and a stable error envelope.

Use PostgreSQL with SQLAlchemy and Alembic for durable relational state, Redis for shared
caches and job/session coordination, Celery for durable background work, Qdrant for
vector retrieval, and SHA-256 content hashes for file-derived cache keys. In-memory
implementations are acceptable only as explicit local/test adapters.

## Implementation completeness

- Never ship placeholder implementations, mock production responses, fabricated model
  output, empty success paths, commented-out intended behavior, or TODO/FIXME comments.
- Tests may use explicit fakes or mocks at external boundaries; production wiring may not.
- If an external dependency cannot be activated in the current environment, implement
  its complete production adapter and configuration boundary, fail clearly when it is
  unavailable, and document the exact activation location and verification command.
  Never report the dependency or phase as complete until the real integration is tested.
- Do not silently fall back from a configured clinical model to heuristics or fabricated
  data. Any approved fallback must be explicit in configuration, response metadata, logs,
  documentation, and tests.

## AI inference contract

Every AI stage must expose a meaningful confidence metric, exact model name and version,
processing time, cache status, and trace identifiers. Never invent confidence values. If
a model has no native probability, use a documented, tested confidence method and expose
the method and calibration version alongside the score.

At minimum, stage results and public responses where applicable include:

- `request_id` and durable `report_id`/processing identifier;
- `model_name` and immutable `model_version` or revision;
- `confidence`, `confidence_method`, and calibration version when calibrated;
- `processing_time_ms` and `cache_hit`;
- pipeline, provider, prompt/rule, and schema versions needed for reproduction;
- warnings and an explicit review/failure state when confidence is insufficient.

Every request must be traceable across API, worker, database, cache, vector store, and
model logs without logging clinical text. Every inference must be reproducible to the
extent supported by its runtime: record exact artifacts and configuration, preprocessing
and prompt versions, random seed, generation parameters, hardware/runtime metadata, and
determinism limitations.

## Engineering rules

- Use Python type hints, focused docstrings, dependency injection, SOLID boundaries, and
  repository abstractions where persistence warrants them.
- Avoid duplicated logic, global mutable state, hard-coded paths, model coupling, and
  configuration outside environment-backed settings.
- Use async I/O where it improves concurrency; keep blocking inference away from the
  event loop.
- Validate file content and size, reject unsupported inputs, handle temporary files
  securely, and never return stack traces or sensitive clinical content in error details.
- Treat uploaded reports, extracted text, prompts, model output, logs, and audio as
  sensitive health data. Minimize retention and logging, enforce tenant boundaries, and
  never use real patient data in tests.
- Cache expensive deterministic work, but include relevant pipeline/model/prompt versions
  in cache identity and follow the configured retention policy.
- Structured logs and metrics should include request ID, pipeline stage, latency,
  processing time, cache hit/miss, model name/version, confidence where meaningful, and
  safe resource metrics. Never log raw clinical text by default.
- Every module requires unit, integration, API, failure-path, and health/readiness tests as
  applicable. Tests must use synthetic or de-identified fixtures.
- A developer interface may be built with React, Vite, and TypeScript for inspecting
  outputs, confidence, latency, logs, downloads, API responses, and Swagger. It is an
  internal tool, not the patient-facing product.

## Quality gate

Before handing off a change, run the relevant tests and static checks, add an Alembic
migration for every schema change, update `ARCHITECTURE.md` when a recorded decision or
boundary changes, update `ROADMAP.md` and `IMPLEMENTATION_LOG.md` when phase status or
implementation history changes, and state any unverified assumptions or unavailable
external services.

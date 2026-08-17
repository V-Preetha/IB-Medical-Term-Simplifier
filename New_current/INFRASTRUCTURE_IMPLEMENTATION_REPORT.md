# Phase 3 + Phase 4 Production Infrastructure Report

Date: 2026-08-05  
Disposition: **SOFTWARE IMPLEMENTED — LIVE DEPLOYMENT VALIDATION PENDING**

## Scope

This change implements the PostgreSQL, SQLAlchemy 2.x, Alembic, Redis, Celery, job API,
health, dashboard, observability, and Docker boundaries requested for Phases 3 and 4.
OCR, Medical NER, Entity Linking, and Relation Extraction packages were not modified.
No Qdrant, verification, translation model, or TTS behavior was introduced.

## Persistence design

The authoritative `DB.pdf`/`ARCHITECTURE.md` schema is preserved rather than replaced by
new names:

| Requested concept | Physical model |
| --- | --- |
| Documents | `reports` |
| OCR results | `report_processing.raw_extracted_text` plus `model_outputs` |
| NER results | `medical_entities` plus `model_outputs` |
| Entity Linking results | additive `entity_links` |
| Embeddings metadata | additive `embedding_records`; raw vectors are not stored |
| Simplifications | authoritative `simplifications` |
| Translations | additive `translations` |
| Processing jobs | additive `processing_jobs` |
| Audit logs | additive append-only `audit_logs` |
| Model registry | additive `model_registry` |

All 17 tables use UUID primary keys and the shared `created_at`, `updated_at`, and
optimistic `version` columns. User-visible aggregates use soft deletion where appropriate;
processing history and audit rows remain visible for traceability. `reports` stores an
encrypted object-storage reference and SHA-256 identity, never report bytes. Embedding
records store metadata only. The initial migration upgrades an empty PostgreSQL database
and has a complete reverse-order downgrade.

## Redis and job behavior

`RedisStageCache` encrypts canonical JSON with a deployment-owned Fernet key. Keys include
tenant, document hash, stage, pipeline, model, configuration, prompt/rule, and schema
versions. Configurable TTL, invalidation, metadata, cache statistics, and token-checked
single-flight locks are implemented. Redis outages return typed 503 failures.

`ProcessingJob` records ordered stage state, request/report identity, progress, retries,
configuration/model versions, Celery identity, safe errors, cancellation, and timestamps.
Duplicate active submissions share a SHA-256 idempotency key and return the existing job.
The database commit precedes broker submission. If submission fails, the durable row moves
to `retrying`; Celery Beat periodically resubmits it after broker recovery.

Celery tasks use JSON only, late acknowledgement, worker-loss rejection, bounded timeouts,
exponential backoff for explicitly transient failures, terminal handling for permanent
failures, CPU/GPU queues, one-prefetch execution, and cancellation checks. Named tasks
exist for all six required stages. Stage implementation is discovered through a stable
entry-point contract so infrastructure does not import or alter frozen AI providers.

## API and dashboard

- `POST /api/v1/jobs` — create an idempotent durable job (`202`).
- `GET /api/v1/jobs/{job_id}` — retrieve tenant-scoped state.
- `GET /api/v1/jobs` — list recent tenant-scoped jobs with status filtering.
- `DELETE /api/v1/jobs/{job_id}` — cancel, soft-delete, and safely revoke (`202`).
- `GET /api/v1/infrastructure/health` — PostgreSQL, Redis, Celery, migration, pool,
  queue, cache, and job metrics.
- `/infrastructure` — Jinja2/Bootstrap 5/vanilla-JavaScript engineering dashboard.

Endpoints have versioned typed schemas, summaries, descriptions, examples, documented
status codes, request IDs, and stable error envelopes. Logs contain request/job IDs,
stage, lifecycle event, and safe errors without clinical text.

## Docker topology

`docker-compose.yml` defines PostgreSQL 16, Redis 7 with append-only persistence, a
one-shot Alembic migration service, FastAPI, Celery worker, and Celery Beat. PostgreSQL and
Redis use persistent named volumes and health checks. Secrets and approved model/resource
mounts are required through environment variables; no production password or encryption
key is committed.

## Verification evidence

- SQLAlchemy metadata imports with 17 expected tables.
- Alembic identifies exactly one head: `0001_initial_schema`.
- PostgreSQL offline upgrade SQL generation passed and ends by inserting that revision.
- PostgreSQL offline downgrade SQL generation passed through complete table/enum removal.
- Repository tests pass tenant isolation, optimistic versioning mechanics, and soft delete.
- Job service tests pass durability, duplicate suppression, queue submission, and cancel.
- Redis tests pass version/tenant key isolation, encrypted storage, TTL metadata, read,
  and deletion using the protocol-compatible in-memory Redis test adapter.
- Celery tests verify the six stage tasks, recovery task, JSON policy, durable delivery
  settings, routes, retry configuration, and Beat schedule.
- API, OpenAPI, health, static assets, and dashboard tests pass.
- Docker Compose YAML parses with API, PostgreSQL, Redis, migration, worker, and Beat
  services plus both persistent volumes.

## Validation limitations

This Windows host has no Docker CLI, PostgreSQL server/client, or Redis server executable.
Consequently, a live PostgreSQL upgrade/downgrade, real Redis expiry/lock Lua behavior,
real worker restart/recovery, and compose health convergence are **NOT VERIFIED** here.
Protocol-compatible SQLite/fakeredis tests and PostgreSQL offline DDL generation are valid
software evidence but are not substitutes for deployment tests.

The frozen AI packages do not currently publish the six `ib_health.pipeline_stages`
entry points. Infrastructure therefore fails closed if a real stage job is submitted in
this workspace; binding those existing application services requires a separately reviewed
cross-phase integration after the infrastructure deployment is live. No empty task success
or fabricated model output is used.

## Recommendation

Deploy the compose topology in an approved integration environment, run Alembic upgrade
and downgrade against PostgreSQL, exercise Redis TTL/lock/outage behavior, start worker and
Beat processes, register reviewed adapters for the frozen stages, and execute restart,
duplicate, retry, cancellation, and failure-recovery scenarios. Until those objective
checks pass, Phases 3 and 4 remain `IN PROGRESS` rather than being marked complete.

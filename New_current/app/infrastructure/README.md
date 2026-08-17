# Production Infrastructure

This package implements the Phase 3 and Phase 4 boundary without importing model SDKs
or changing OCR, Medical NER, Entity Linking, or Relation Extraction code.

## Runtime topology

`InfrastructureRuntime` owns one async SQLAlchemy engine/session factory, one Redis client,
one encrypted stage cache, one Celery application, and one transactional job service.
FastAPI exposes `/api/v1/jobs`, `/api/v1/jobs/{job_id}`, and
`/api/v1/infrastructure/health`. Celery exposes named tasks for OCR, NER, Entity Linking,
Embeddings, Simplification, and Translation plus a periodic recovery task for jobs that
were committed while the broker was unavailable.

Stage tasks discover deployment integrations through the
`ib_health.pipeline_stages` Python entry-point group. Each entry-point name is the stage
value (`ocr`, `ner`, `entity_linking`, `embeddings`, `simplification`, or `translation`)
and resolves to an async callable accepting `StageContext`. This preserves the frozen AI
packages: infrastructure never instantiates a concrete model provider. A missing or
ambiguous executor is a terminal configuration failure, not an empty success response.

## Required configuration

- `INFRASTRUCTURE_ENABLED=true`
- `DATABASE_URL=postgresql+asyncpg://...`
- `REDIS_URL=redis://...` or `rediss://...`
- `CACHE_ENCRYPTION_KEY=<Fernet key>`
- `CELERY_BROKER_URL` and `CELERY_RESULT_BACKEND`
- `CACHE_TTL_SECONDS`, `CACHE_LOCK_TTL_SECONDS`, and `REDIS_KEY_PREFIX`
- `CELERY_TASK_TIMEOUT_SECONDS`, `CELERY_MAX_RETRIES`, and
  `CELERY_RETRY_BACKOFF_SECONDS`

Generate a cache key through the deployment secret-management workflow. For local setup,
a valid Fernet key can be produced with:

```powershell
.\.venv\Scripts\python.exe -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Never commit the generated key or a database password. Redis values containing stage
output are encrypted before storage. Cache identity includes tenant ID, document SHA-256,
stage, pipeline version, immutable model revision, configuration version, prompt/rule
version, and schema version. TTLs are mandatory.

## Migrations and processes

```powershell
$env:DATABASE_URL = "postgresql+asyncpg://ibhealth:<password>@localhost:5432/ibhealth"
.\.venv\Scripts\alembic.exe upgrade head
.\.venv\Scripts\celery.exe -A app.infrastructure.celery_app:celery_app worker --loglevel=INFO --queues=ibhealth.cpu,ibhealth.gpu --concurrency=1
.\.venv\Scripts\celery.exe -A app.infrastructure.celery_app:celery_app beat --loglevel=INFO
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Use `docker compose up --build` after copying `.env.infrastructure.example` to `.env`,
supplying secrets, and mounting approved model/post-processing resources. The compose
topology runs PostgreSQL, Redis, the migration job, API, worker, and optional Beat service
with persistent database/cache volumes and dependency health checks.

The local process default is `INFRASTRUCTURE_ENABLED=false`; this keeps model-only
engineering work independent of unavailable external services. In that state the health
endpoint returns HTTP 503 with `not_configured`, and job mutations fail closed.


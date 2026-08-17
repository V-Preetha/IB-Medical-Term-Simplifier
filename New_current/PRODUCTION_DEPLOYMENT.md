# Production deployment for asynchronous report processing

## API contract

The existing synchronous `POST /process-report` endpoint and response schema are
unchanged. New web/mobile clients should submit long-running work through:

```text
POST /reports
GET  /reports/{job_id}/status
GET  /reports/{job_id}/result
```

Submission and pending result responses use HTTP 202. Clients should poll after one
second, increase to two seconds after five attempts, cap at five seconds, stop after a
terminal `completed` or `failed` state, and resume polling after app suspension using
the persisted `job_id`.

## Queue topology

The repository implementation uses FastAPI BackgroundTasks, a bounded job registry,
single-flight duplicate suppression, and the existing process-local model instances.
It is suitable for one inference service process, including one process pinned to one
GPU. Jobs and completed result metadata expire after the configured TTL and do not
survive a process restart.

For restart-safe or horizontally scaled production, preserve the HTTP schemas and
replace the process-local job manager with:

```text
Web/mobile client
        |
        v
Stateless FastAPI replicas
        |
        +--> Redis: Celery broker, job state, progress, normalized-hash index
        |
        +--> encrypted object storage: uploads and completed JSON
                    |
                    v
          Celery GPU inference workers
          (one process per GPU)
```

Recommended Celery settings:

- route report jobs to a dedicated `medical-reports-gpu` queue;
- use `worker_concurrency=1` per GPU to avoid duplicated model memory;
- acknowledge after completion and use visibility timeouts longer than the maximum
  report runtime;
- make the normalized SHA-256 cache key idempotent before acknowledging;
- publish progress updates no more than two to four times per second;
- store uploads/results outside Redis and use short-lived encrypted object references;
- configure retry only for transient infrastructure/model availability failures, not
  invalid documents;
- use TTLs and deletion policies appropriate for medical data.

Do not run multiple Uvicorn workers against one GPU unless each worker has enough VRAM
for its own GLiNER, ModernBERT, and Qwen copies. Prefer one inference worker per GPU and
scale API replicas independently.

## GPU inference recommendations

- Use NVIDIA L4, A10, A100, or equivalent with sufficient VRAM for all configured
  models and KV cache.
- Install a CUDA-enabled PyTorch build and verify startup logs report
  `qwen_device=cuda`.
- Pin workers with `CUDA_VISIBLE_DEVICES`; fail readiness if models cannot load.
- Prefer BF16 on supported GPUs, otherwise FP16.
- Install Flash Attention 2 when compatible; otherwise SDPA remains enabled.
- Warm one representative request after deployment so CUDA graphs/compilation and
  allocator pools are ready before traffic.
- Export queue depth, job age, stage time, CUDA utilization, VRAM, prompt/output tokens,
  tokens per second, cache hit ratio, failure rate, and polling request rate.

## Latency service targets

These are deployment targets, not guarantees; validate them on the selected GPU and
representative medical reports:

| Operation | Target |
|---|---:|
| Job acknowledgement after upload | p95 below 250 ms |
| Status/result polling | p95 below 100 ms |
| Exact-upload cache hit | p95 below 250 ms |
| Normalized-result hit after extraction | p95 below 2 s for digital text |
| 1-page digital report on GPU | p95 3–8 s |
| 4-page digital report on GPU | p95 8–20 s |
| 10-page digital report on GPU | p95 15–45 s |

OCR-heavy reports vary with render resolution, page complexity, and handwriting and
should remain asynchronous regardless of estimated duration. Queue wait time must be
reported separately from processing time.

## Cache and privacy

The application hashes UTF-8 normalized text with SHA-256 and caches only completed
structured clinical output. Concurrent requests for the same normalized hash use a
single-flight lock so only one performs GLiNER/Qwen inference.

For distributed deployment, use Redis for the normalized-hash index and encrypted
object storage for result bodies. Include model/prompt version in the distributed key
when those components change, enforce tenant isolation, never expose hashes as public
identifiers, and apply auditable retention/deletion policies.

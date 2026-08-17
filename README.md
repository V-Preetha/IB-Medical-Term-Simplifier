# Medical Term Simplifier

Converts medical documents — lab reports, discharge summaries, prescriptions, and scanned
reports — into medically faithful, patient-friendly explanations at three reading levels,
with optional translation into Indian regional languages.

**Status: active prototype, not production-ready.** See
[`PROJECT_PROGRESS_REPORT_JUNE_AUGUST_2026.md`](PROJECT_PROGRESS_REPORT_JUNE_AUGUST_2026.md)
for the full narrative of what has been built and validated, and [`ROADMAP.md`](ROADMAP.md)
for the phase-by-phase engineering contract this repository follows.

## Overview

A patient uploads a medical document. The system extracts its text (including from scans and
handwriting), identifies medical entities, rewrites the content into Clinical / General
Public / Child-Friendly explanations without altering the underlying facts, and — optionally
— translates the result into a supported Indian language.

## Features

- OCR / document understanding for PDF, PNG, JPEG, and TIFF, including a fast path for
  digitally-native PDFs
- Medical named-entity recognition (diseases, medications, procedures, anatomy, lab values)
- Three-level report simplification with a fail-closed factual-grounding check — output that
  introduces an unsupported medical fact is rejected rather than shipped
- Translation to Hindi, Tamil, and Kannada with numeric/unit/dosage values protected across
  the translation boundary
- Per-service engineering dashboards and a consolidated pipeline console
  (`/engineering-demo`) for manually exercising the whole chain

## Current Architecture

The production MVP flow is intentionally narrower than the full target architecture:

```text
Upload → OCR (Qwen3-VL) → Medical NER → Qwen3 Simplification → IndicTrans2 Translation
```

Entity Linking, Relation Extraction, Medical Verification, and Text-to-Speech exist as
architecture (typed contracts, providers, routes, health checks) but are **deliberately
deferred** — none of them run in the MVP request path. Medical embeddings run only as an
optional, non-blocking background operation.

| Layer | Current model / tool | Status |
|---|---|---|
| OCR / Vision | `Qwen/Qwen3-VL-4B-Instruct` | Implemented, runtime-validated on CPU and GPU |
| OCR cleanup | SymSpell + regex + medical abbreviation dictionary | Implemented |
| Medical NER | `d4data/biomedical-ner-all` | Implemented, runtime-validated |
| Entity linking | SciSpaCy + UMLS | Architecture only — no approved model or UMLS license yet |
| Relations | `michiyasunaga/BioLinkBERT-base` | Backbone pinned, artifact is an untrained base encoder — fails closed |
| Embeddings | BioClinical ModernBERT | Provider implemented; exact checkpoint `PENDING_APPROVAL` |
| Vector retrieval | Qdrant | Not implemented |
| Simplification | `Qwen/Qwen3-0.6B` | Implemented, runtime-validated |
| Verification | `pritamdeka/PubMedBERT-MNLI-MedNLI` | Technically validated; license pending, not production-approved |
| Translation | `ai4bharat/indictrans2-en-indic-dist-200M` | Implemented, runtime-validated end-to-end |
| TTS | Kokoro TTS (target) | Not implemented — deferred for MVP |
| API | FastAPI | Implemented |
| Database | PostgreSQL (async SQLAlchemy + Alembic) | Fully coded; never run against a live PostgreSQL instance |
| Cache | Redis | Fully coded; never run against a live Redis instance |
| Background tasks | Celery | Fully coded; never run against a live broker |

Exact repository IDs, pinned revisions, licenses, and checksums for every model are the
authoritative content of [`MODEL_MANIFEST.md`](MODEL_MANIFEST.md) — treat this README's table
as a summary, not the source of truth.

## Current Implementation Status

- **Working, with real local model inference:** OCR, Medical NER, Simplification, and
  Translation — including one real, no-mock, end-to-end run through all four stages.
- **Fully coded but unvalidated against live infrastructure:** PostgreSQL, Redis, Celery.
- **Partially implemented:** Medical Verification (technically working, not license-cleared),
  Medical Embeddings (works standalone, not connected to any retrieval store).
- **Not implemented:** Entity Linking runtime, Relation Extraction runtime, Text-to-Speech,
  Qdrant-based knowledge retrieval.
- A PP-OCRv6 candidate OCR model was evaluated and **correctly rejected** in August 2026
  after its recognition-model metadata was found to be internally contradictory
  (`ARTIFACT_IDENTITY_AMBIGUOUS`) — it is not in the approved model manifest and is not used.

See the progress report for the full timeline and evidence behind each of these claims.

## Technology Stack

FastAPI, SQLAlchemy 2.x + Alembic, Redis, Celery, PyTorch + Transformers, Jinja2 + Bootstrap 5
+ vanilla JS for the engineering dashboards, pytest + Ruff for testing and linting, Docker /
Docker Compose for the (unvalidated) deployment topology.

## Installation

The installable package lives in `New_current/`.

```bash
cd New_current
python -m pip install -e ".[dev]"
```

Optional dependency groups (`entity-linking`, `clinical-ner`, `translation`) install the
extra packages needed for those specific providers — see `New_current/pyproject.toml`.

## Running the API

```bash
cd New_current
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The engineering dashboard is served at `/`, Swagger UI at `/docs`, and OpenAPI at
`/openapi.json`. Every service also exposes a `GET /api/v1/<service>/health` endpoint;
providers report `not_configured` or `degraded` rather than silently proceeding when an
approved model checkpoint is missing.

## API Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/ocr` | Document upload and OCR |
| `POST /api/v1/ner` | Medical entity recognition |
| `POST /api/v1/simplify` | Three-level report simplification |
| `POST /api/v1/translations` | Translation of simplified output |
| `POST /api/v1/verification` | Factual entailment check (technical validation only) |
| `POST /api/v1/embeddings` | Medical text embeddings (background use only) |
| `POST /api/v1/entity-linking` | Entity linking (`not_configured` until UMLS is provisioned) |
| `POST /api/v1/relation-extraction` | Relation extraction (fails closed — incompatible artifact) |
| `GET /api/v1/runtime/metrics` | Real-time GPU/CPU memory and per-stage timing |
| `GET/POST /api/v1/jobs` | Durable job submission and status (Celery-backed) |

## Models

See [`MODEL_MANIFEST.md`](MODEL_MANIFEST.md) for the authoritative, machine-readable list of
approved model repository IDs, pinned revisions, licenses, and checksums. Nothing in this
repository downloads a model at a moving revision (`latest`, `main`, etc.) — every provider
fails closed until an exact approved identity is configured.

## Hardware / Runtime Expectations

Development and validation were performed on a machine with an NVIDIA RTX 5050 Laptop GPU
(8 GB VRAM). All providers default to `device=auto` and run correctly on CPU, but OCR and
Simplification are substantially slower without a GPU (see the progress report for measured
before/after latencies). `requirements.txt` and `pyproject.toml` resolve to the CPU PyTorch
wheel by default; installing a CUDA build is a separate, host-specific step, not something
this repository pins automatically.

## Project Structure

```text
.
├── New_current/            # Production source boundary — the FastAPI application
│   ├── app/                # One vertical slice per service (ocr, ner, simplification, ...)
│   ├── tests/               # pytest suite (118 tests as of this writing)
│   ├── benchmarks/          # Offline validation/benchmark harnesses per service
│   └── migrations/          # Alembic migrations
├── ARCHITECTURE.md          # Full target architecture and DB schema source of truth
├── ROADMAP.md                # Phase-by-phase engineering contract and completion criteria
├── MODEL_MANIFEST.md         # Approved model inventory (exact IDs, revisions, licenses)
├── IMPLEMENTATION_LOG.md     # Dated, evidence-based build log
├── docs/internship_submission/  # Internship-specific reporting, not architecture docs
├── medical_term_sft/         # QLoRA fine-tuning pipeline (dataset + training script; no
│                                trained adapter exists yet — see Known Limitations)
├── sft_data_pipeline/        # Synthetic instruction-tuning dataset generation (~5,000 examples)
└── Evaluation/                # Standalone evaluation/benchmarking scripts from the June baseline
```

## Evaluation

An offline benchmark harness exists at `New_current/benchmarks/clinical_performance/`, and
per-service benchmark tooling exists under `New_current/benchmarks/{ocr,ner,translation}/`.
None of these have yet been run against a representative, clinician-reviewed corpus — current
evidence is limited to small synthetic datasets and single-request smoke tests. A proper
evaluation suite (CER/WER for OCR, precision/recall/F1 for NER, factual-preservation and
readability scoring for Simplification, entailment accuracy for Verification, and
terminology-preservation review for Translation) is planned but not yet built.

## Known Limitations

- No representative, clinician-reviewed evaluation corpus exists for any stage yet.
- PostgreSQL, Redis, and Celery are fully implemented but have never been exercised against
  live services — persistence and job durability are unverified.
- Entity Linking, Relation Extraction, and Text-to-Speech have architecture but no runnable
  production model.
- Medical Verification works technically but its model's license has not been cleared, so it
  is not production-approved.
- The QLoRA fine-tuning pipeline (`medical_term_sft/`) has a dataset and training script, but
  the one training run attempted was interrupted at epoch 1 of 3 — **no trained adapter
  exists**. Production simplification uses the base `Qwen/Qwen3-0.6B` model, not a fine-tune.
- `Evaluation/test.py` previously contained a hardcoded Hugging Face token; it is excluded
  from version control via `.gitignore` and must be scrubbed/rotated before that file is ever
  tracked.
- Performance figures throughout the documentation are single-request smoke tests on one
  development machine, not load-tested production benchmarks.

## Roadmap

See [`ROADMAP.md`](ROADMAP.md) for the full 14-phase engineering contract. Immediate
priorities: clinical validation of OCR/NER/Simplification against a real reviewed corpus,
live PostgreSQL/Redis/Celery validation, and resolving Entity Linking's UMLS licensing.

## License / Model Licensing Notes

This repository is licensed under Apache-2.0 (see [`LICENSE`](LICENSE)). That license covers
this project's own code — it does **not** relicense the third-party models it uses. Each
model in `MODEL_MANIFEST.md` carries its own upstream license (Apache-2.0, MIT, or, for
Medical Verification, a license that has not yet been confirmed and is marked
`PENDING_VERIFICATION`). Medical knowledge resources referenced in the roadmap — UMLS,
SNOMED CT, ICD-10, MedlinePlus, PubMed data, DrugBank — each carry their own access and
redistribution terms; none should be assumed freely redistributable without separately
confirming their license.

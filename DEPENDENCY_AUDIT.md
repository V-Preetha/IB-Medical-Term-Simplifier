# Phase 2 OCR Dependency Audit

Audit date: 2026-08-04  
Production boundary: `New_current/`  
Architecture status: frozen

This audit covers the Phase 2 OCR runtime, its validation tooling, and its approved model
artifacts. PostgreSQL, Alembic, Redis, and Celery remain outside Phase 2. The OCR runtime
does not require a system OCR executable or an alternative OCR engine.

## Python runtime packages

| Dependency | Why it exists | Module or surface | Mandatory | Approved architecture |
| --- | --- | --- | --- | --- |
| FastAPI | Typed REST API, dependency injection, OpenAPI, health endpoints | `app.main`, `app.api` | Yes | Yes |
| Jinja2 | Same-origin internal engineering dashboard templates | `app.main`, `app/templates` | Yes | Yes |
| Pillow | Image validation, decoding, frame handling, EXIF orientation, resizing | `app.ocr.providers.documents` | Yes | Yes |
| pillow-heif | Registers HEIC/HEIF decoding used by the accepted upload contract | `app.ocr.providers.documents` | Yes for HEIC support | Yes |
| PyMuPDF | Validates, decrypt-checks, page-counts, and renders PDFs | `app.ocr.providers.documents` | Yes | Yes |
| python-multipart | Parses uploaded multipart document bodies | FastAPI upload routes | Yes | Yes |
| PyTorch | Tensor execution, device selection, confidence calculations, GPU metrics | Qwen3-VL provider | Yes for live inference | Yes |
| Transformers | Loads the pinned processor and model class | Qwen3-VL provider | Yes for live inference | Yes |
| Uvicorn and standard extras | ASGI production process and HTTP runtime | Service startup | Yes for deployed process | Yes |

The regex normalizer, medical abbreviation stage, and bounded SymSpell implementation
are repository code and versioned data files; they do not depend on a second OCR engine
or an external spelling executable.

## Validation and development packages

| Dependency | Why it exists | Module or surface | Mandatory | Approved architecture |
| --- | --- | --- | --- | --- |
| HTTPX | ASGI/API integration and OpenAPI tests | `tests` | Development only | Yes |
| psutil | Process RSS measurements in benchmark evidence | `benchmarks/ocr/run_validation.py` | Validation only | Yes |
| pytest | Unit, integration, API, failure-path, and health tests | `tests` | Development only | Yes |
| Ruff | Lint and import-order quality gate | Repository quality gate | Development only | Yes |

The optional `clinical-ner` dependency group (`gliner`, PyTorch, and Transformers) belongs
to a future pipeline phase. It is not imported by or required to run the Phase 2 OCR
service and is not Phase 2 completion evidence.

## System packages and external executables

| Dependency | Why it exists | Module or surface | Mandatory | Approved architecture |
| --- | --- | --- | --- | --- |
| NVIDIA display driver | Makes a supported NVIDIA device available to CUDA-enabled PyTorch | Optional CUDA inference | No; CPU is the required baseline | Yes |
| CUDA runtime compatible with PyTorch | Enables optional CUDA model execution | PyTorch providers | No; CUDA validation is conditional | Yes |
| `nvidia-smi` | Read-only physical GPU inventory for validation evidence | Validation runner | No | Yes, validation telemetry only |

There are no mandatory external executables for the approved OCR pipeline. A browser is
used only to capture Swagger and dashboard evidence and is not a service runtime
dependency. PostgreSQL, Redis, Celery workers/brokers, and their executables are excluded
until their roadmap phases.

## Models and versioned language resources

| Artifact | Why it exists | Module or surface | Mandatory | Approved architecture |
| --- | --- | --- | --- | --- |
| Qwen3-VL checkpoint | Page-ordered visual OCR and prompted document-type inference | `Qwen3VLOCRProvider` | Yes for production inference | Yes; immutable identity and revision are pinned in `MODEL_MANIFEST.md` |
| Medical abbreviation dictionary | Versioned abbreviation expansion | `MedicalAbbreviationStage` | Yes | Yes |
| SymSpell frequency dictionary | Deterministic spelling correction | `SymSpellStage` | Yes | Yes |
| Protected medical terms | Prevents unsafe corrections of clinical tokens | Post-processing provider | Yes | Yes |

No checkpoint may be downloaded from a moving revision. Live inference remains fail-closed
until `MODEL_MANIFEST.md` or the corresponding approved environment variables provide an
exact repository ID and immutable revision, and the local artifact is available.

## Dependency conclusion

Every mandatory Phase 2 runtime package maps to the frozen architecture. No alternative
OCR engine, unapproved model, database client, cache client, or task queue is present in
the production package manifest. Model checkpoint approval and live validation remain
open gates; their absence is `NOT VERIFIED`, not a dependency failure.

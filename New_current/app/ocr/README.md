# OCR service package

This package is the sole executable OCR boundary for the versioned OCR service.

Dependency direction is inward:

```text
api -> application -> domain
                    -> provider contracts
infrastructure ----> provider contracts
providers ---------> provider contracts
postprocessing ----> domain
```

Package responsibilities:

- `api`: FastAPI routes, dependency wiring, and public request/response translation.
- `application`: OCR orchestration, lifecycle policy, and use cases.
- `domain`: infrastructure-independent OCR entities, values, and state.
- `providers`: replaceable multimodal OCR and post-processing provider contracts/adapters.
- `postprocessing`: regex, abbreviation, and SymSpell processing behind typed boundaries.
- `infrastructure`: persistence, cache, job, and external-service adapters.
- `observability`: safe structured logging, metrics, and runtime metadata.

Persistence is represented only by the asynchronous repository and unit-of-work ports in
`application/repositories.py`. Phase 2 does not provide a database adapter, SQLAlchemy
model, or migration. The Phase 3 adapter must map these tenant-scoped contracts onto the
authoritative `reports`, `report_processing`, and `model_outputs` schema.

`OCRApplicationService` is the only orchestration path. Routes call that service, the
service depends on provider contracts, and provider instances are supplied through the
configuration-driven registry and factories. Routes and application services do not
import model libraries directly.

## Provider configuration and discovery

Provider selection is required through `OCR_PROVIDER` and `POSTPROCESSOR_PROVIDER`.
Provider-specific values are loaded without code changes from variables prefixed by
`OCR_CONFIG__` and `POSTPROCESSOR_CONFIG__`; the suffix becomes the provider's lowercase
configuration key. Built-in provider names are `qwen3-vl` and `symspell`.

External packages can register providers through these entry-point groups:

- `ib_health.ocr.providers`
- `ib_health.ocr.postprocessors`

Configuration values whose names indicate credentials, keys, passwords, secrets, or
tokens are redacted from metadata and health responses. The Qwen OCR prompt is also
redacted while its version remains visible.

## Model activation

Install the project dependencies and approve the exact model identities in the repository
`MODEL_MANIFEST.md` or supply the corresponding environment variables. Qwen3-VL performs
OCR and document-type inference in one versioned structured generation.

Qwen3-VL requires these `OCR_CONFIG__` suffixes:

- `PROVIDER_VERSION`, `MODEL_NAME`, immutable `MODEL_REVISION`, and `HF_CACHE_DIR`
- `LOCAL_FILES_ONLY`, `DEVICE`, `ALLOW_CPU_FALLBACK`, and `DTYPE`
- `BATCH_SIZE`, `MAX_IMAGE_SIZE`, `MAX_PAGES`, `PDF_RENDER_DPI`, and `TIMEOUT_SECONDS`
- `MAX_NEW_TOKENS`, `DO_SAMPLE`, `TEMPERATURE`, `TOP_P`, `NUM_BEAMS`, and `SEED`
- `PROMPT`, immutable `PROMPT_VERSION`, `CONFIDENCE_THRESHOLD`, and
  `CONFIDENCE_CALIBRATION_VERSION`

The medical post-processor requires these `POSTPROCESSOR_CONFIG__` suffixes:

- `PROVIDER_VERSION`, `RULE_VERSION`, `DICTIONARY_PATH`, and `DICTIONARY_VERSION`
- `ABBREVIATION_DICTIONARY_PATH`, `ABBREVIATION_DICTIONARY_VERSION`, and
  `PROTECTED_TERMS_PATH`
- `MAX_EDIT_DISTANCE`, `PREFIX_LENGTH`, `MIN_WORD_LENGTH`, and `MIN_FREQUENCY`

Provider construction is lazy. FastAPI lifecycle initialization loads each configured
model once, health becomes `ready` only after successful loading, requests reuse that
instance under an inference lock, and shutdown releases model references and CUDA cache.
Missing manifest approval, model artifacts, dictionaries, or CUDA without an approved CPU
fallback fail closed.

Run the provider verification suite without downloading models:

```powershell
cd New_current
.\.venv\Scripts\python.exe -m pytest -q tests\ocr
```

The tests use synthetic documents and explicit fakes only at model-library boundaries.
To verify real deployment artifacts, configure the environment above and start
`uvicorn app.main:app`; `GET /api/v1/ocr/health` must report every provider as `ready`
before inference traffic is accepted.

# IB Health OCR Service

`New_current/` is the production source boundary for Phase 2. The service implements the
frozen OCR pipeline:

```text
Upload -> Qwen3-VL OCR and prompted document-type inference
       -> Regex normalization -> Medical abbreviation dictionary -> SymSpell
       -> versioned API response
```

The FastAPI routes depend on `OCRApplicationService`. That service obtains providers only
through typed contracts, registry-backed factories, and FastAPI dependency injection.
The database, Redis, and background-worker adapters remain interfaces until their roadmap
phases.

## Model approval

Model repository identifiers and immutable revisions come exclusively from the repository
root `MODEL_MANIFEST.md` or the corresponding deployment environment variables. There are
no implicit model identities and no moving-revision defaults. Startup fails closed until
the approved model identity is supplied and its checkpoint is available.

Required provider selection variables:

```text
OCR_PROVIDER=qwen3-vl
POSTPROCESSOR_PROVIDER=symspell
```

Model identity overrides, when approved for a deployment:

```text
OCR_CONFIG__MODEL_NAME
OCR_CONFIG__MODEL_REVISION
OCR_CONFIG__HF_CACHE_DIR
```

All other provider settings and required post-processing resource paths are documented in
`app/ocr/README.md`. Uploaded clinical content and extracted text are not written to normal
logs.

## Running locally

From this directory, install the package and start the ASGI service:

```powershell
python -m pip install -e ".[dev]"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The engineering dashboard is served at `/`, Swagger at `/docs`, OpenAPI at
`/openapi.json`, and OCR operations under `/api/v1/ocr`. Provider readiness will remain
unavailable until approved checkpoints and complete provider configuration are present.

## Quality and validation

```powershell
python -m ruff check app tests benchmarks
python -m pytest -q
python -m compileall -q app tests benchmarks
python benchmarks/ocr/run_validation.py
```

The validation runner reads `MODEL_MANIFEST.md`, does not download models, and emits
Markdown, JSON, CSV, and structured-log evidence beneath `benchmarks/ocr/reports/`.
Unavailable approvals or checkpoints are recorded as `NOT VERIFIED`; they are never
substituted with another model.

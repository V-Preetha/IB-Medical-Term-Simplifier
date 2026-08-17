# IndicTrans2 MVP Translation

The production adapter is local-only and fail-closed. It never downloads a model or uses a
heuristic/remote fallback.

The approved production candidate is `ai4bharat/indictrans2-en-indic-dist-200M` revision
`173b94239f7c38886b2747b8d4a5db771a7e1232`, under the MIT license. The local checkpoint
is provisioned at the exact manifest path. Its approved primary artifact is
model.safetensors with SHA-256
0039e4304e9889acc5c8350a193311d07aa5399b6d9fb0445e8fde19e3533bb5. Do not use a branch
name, latest, a substituted checkpoint, or pytorch_model.bin.

After that approved snapshot is provisioned, set these deployment values exactly:

```powershell
$env:TRANSLATION_CONFIG__MODEL_ID = "ai4bharat/indictrans2-en-indic-dist-200M"
$env:TRANSLATION_CONFIG__MODEL_PATH = "E:\Internships\Icebrkr\Medical-Term-Simplifier\New_current\.model-cache\translation\indictrans2-en-indic-dist-200M"
$env:TRANSLATION_CONFIG__MODEL_REVISION = "173b94239f7c38886b2747b8d4a5db771a7e1232"
$env:TRANSLATION_CONFIG__DEVICE = "auto"
$env:TRANSLATION_CONFIG__MAX_NEW_TOKENS = "256"
```

Install the declared runtime extra with
`.venv\Scripts\python.exe -m pip install -e ".[translation]"`. Then start the service and
verify `GET /api/v1/translations/health`, `GET /api/v1/translations/models`, and a
synthetic/de-identified `POST /api/v1/translations` request. The provider checks the
manifest identity, exact local checkpoint directory, and approved model.safetensors
checksum before loading.
device=auto selects CUDA when the installed PyTorch runtime exposes it; otherwise it uses
CPU. The exact snapshot completed local CUDA runtime validation on 2026-08-09. Clinical
translation-quality validation remains open, so Phase 11 remains in progress.

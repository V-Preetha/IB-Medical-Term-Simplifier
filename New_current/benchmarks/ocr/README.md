# OCR validation and benchmarking

`run_validation.py` is a fail-closed evidence runner for Phase 2 Step 4. It never selects
or downloads a checkpoint. Configure the production provider environment documented in
`app/ocr/README.md`, then run:

```powershell
.\.venv\Scripts\python.exe benchmarks\ocr\run_validation.py
```

The runner attempts CPU and CUDA validation independently, measures model loading, first
and warm inference, pages per second, process RSS, provider GPU memory, and live confidence
distribution, and writes Markdown, CSV, JSON, JSONL, and correction reports beneath
`benchmarks/ocr/reports/`.

Missing provider configuration, approved model artifacts, CUDA support, or ground truth
produces `NOT VERIFIED` and null/absent measurements. Test doubles are never used. The
included generated inputs and post-processing examples are synthetic and de-identified.

# Standalone Medical Report Simplification Pipeline

Fresh local-only implementation; no API or web code.

1. Install: `pip install -r requirements.txt`
2. Use the approved OCR service for image-only inputs.
3. Edit only `INPUT_FILE` in `simplify_report.py`.
4. Run: `python simplify_report.py`

The first run downloads large Hugging Face models and may need substantial RAM/VRAM and disk. Supported files: PDF, scanned PDF, PNG, JPG/JPEG, TXT. Outputs are created in `outputs/` as simplified TXT, structured JSON, and Markdown. Context vectors are used internally but omitted from JSON; model and dimensions are recorded.

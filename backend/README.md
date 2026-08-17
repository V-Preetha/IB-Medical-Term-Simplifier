# AI Medical Report Simplifier Backend

Stage 11 implements the full backend pipeline: parsing, OCR, section
segmentation, SciSpaCy entity recognition, ModernBERT difficult-term detection,
BioClinicalBERT/OpenMed semantic understanding, fusion, Qwen3 simplification,
Granite Guardian validation, and evaluation.

## Run locally

```bash
python -m venv .venv
pip install -r requirements.txt
uvicorn main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

Run the complete pipeline from JSON text:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/reports/simplify \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"Diagnosis: Hypertension\"}"
```

Run the complete pipeline from a PDF, scanned PDF, image, or text file:

```bash
curl -F "file=@report.pdf" http://127.0.0.1:8000/api/v1/reports/simplify/file
```

The final response includes the simplified report, highlighted difficult terms,
term explanations, aggregate confidence, validation status, and evaluation
scores.

Extract report text from a PDF, scanned PDF, image, or text file:

```bash
curl -F "file=@report.pdf" http://127.0.0.1:8000/api/v1/reports/extract
```

Extract directly submitted text:

```bash
curl -F "text=Diagnosis: Hypertension" http://127.0.0.1:8000/api/v1/reports/extract
```

Segment extracted report text:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/reports/segment \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"Diagnosis: Hypertension\nMedications: Lisinopril\"}"
```

Extract medical entities from segmented sections:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/reports/entities \
  -H "Content-Type: application/json" \
  -d "{\"sections\":[{\"section_type\":\"diagnosis\",\"title\":\"Diagnosis\",\"content\":\"Hypertension\",\"order\":0,\"confidence\":0.95}]}"
```

Detect difficult terms with ModernBERT embeddings:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/reports/modernbert/difficult-terms \
  -H "Content-Type: application/json" \
  -d "{\"entities\":[{\"text\":\"Hypertension\",\"entity_type\":\"disease\",\"section_type\":\"diagnosis\",\"section_title\":\"Diagnosis\",\"start_char\":0,\"end_char\":12,\"confidence\":0.9,\"source_label\":\"DISEASE\"}]}"
```

Resolve semantic meaning and clinical context:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/reports/clinical-context \
  -H "Content-Type: application/json" \
  -d "{\"entities\":[{\"text\":\"Hypertension\",\"entity_type\":\"disease\",\"section_type\":\"diagnosis\",\"section_title\":\"Diagnosis\",\"start_char\":0,\"end_char\":12,\"confidence\":0.9,\"source_label\":\"DISEASE\"}]}"
```

Fuse ModernBERT and BioClinicalBERT/OpenMed outputs:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/reports/fusion \
  -H "Content-Type: application/json" \
  -d "{\"difficult_terms\":[{\"term\":\"Hypertension\",\"difficulty\":0.826,\"embedding\":[0.6,0.8],\"confidence\":0.92,\"entity_type\":\"disease\",\"section_type\":\"diagnosis\",\"context\":\"Diagnosis: Hypertension\"}],\"semantic_interpretations\":[{\"term\":\"Hypertension\",\"meaning\":\"High blood pressure.\",\"context\":\"A chronic condition where blood pressure remains higher than normal.\",\"ambiguity_resolution\":\"Resolved as chronic disease.\",\"confidence\":0.945,\"entity_type\":\"disease\",\"section_type\":\"diagnosis\",\"semantic_embedding\":[1.0,0.0],\"matched_concept\":\"hypertension\"}]}"
```

Generate a simplified report from fused structured data:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/reports/simplify/from-fusion \
  -H "Content-Type: application/json" \
  -d "{\"fused_terms\":[{\"term\":\"Hypertension\",\"difficulty\":0.826,\"meaning\":\"High blood pressure.\",\"context\":\"A chronic condition where blood pressure remains higher than normal.\",\"confidence\":0.9393,\"entity_type\":\"disease\",\"section_type\":\"diagnosis\",\"modernbert_confidence\":0.92,\"semantic_confidence\":0.945,\"ambiguity_resolution\":\"Resolved as chronic disease.\",\"modernbert_embedding\":[0.6,0.8],\"semantic_embedding\":[1.0,0.0],\"matched_concept\":\"hypertension\"}]}"
```

Validate a simplification:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/reports/validate \
  -H "Content-Type: application/json" \
  -d "{\"simplified_report\":\"Simplified Report: A patient has hypertension, a chronic condition where blood pressure remains higher than normal. Key Terms: - Hypertension: High blood pressure.\",\"fused_terms\":[{\"term\":\"Hypertension\",\"difficulty\":0.826,\"meaning\":\"High blood pressure.\",\"context\":\"A chronic condition where blood pressure remains higher than normal.\",\"confidence\":0.9393,\"entity_type\":\"disease\",\"section_type\":\"diagnosis\",\"modernbert_confidence\":0.92,\"semantic_confidence\":0.945,\"ambiguity_resolution\":\"Resolved as chronic disease.\",\"modernbert_embedding\":[0.6,0.8],\"semantic_embedding\":[1.0,0.0],\"matched_concept\":\"hypertension\"}]}"
```

Evaluate simplification quality:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/reports/evaluate \
  -H "Content-Type: application/json" \
  -d "{\"reference_text\":\"Hypertension means high blood pressure.\",\"simplified_report\":\"Hypertension means high blood pressure.\",\"fused_terms\":[{\"term\":\"Hypertension\",\"difficulty\":0.826,\"meaning\":\"High blood pressure.\",\"context\":\"A chronic condition where blood pressure remains higher than normal.\",\"confidence\":0.9393,\"entity_type\":\"disease\",\"section_type\":\"diagnosis\",\"modernbert_confidence\":0.92,\"semantic_confidence\":0.945,\"ambiguity_resolution\":\"Resolved as chronic disease.\",\"modernbert_embedding\":[0.6,0.8],\"semantic_embedding\":[1.0,0.0],\"matched_concept\":\"hypertension\"}]}"
```

## Fusion Algorithm

The Stage 7 Fusion Layer uses `weighted-key-match-v1`:

1. Normalize terms by lowercase text, entity type, and section type.
2. Match ModernBERT difficult terms with BioClinicalBERT/OpenMed semantic
   interpretations by full key, then by term-only fallback.
3. Preserve both model outputs in the final structure.
4. Compute fused confidence as `45% ModernBERT confidence + 45% semantic
   confidence + 10% match quality`.
5. Emit warnings for unmatched outputs instead of silently dropping them.

## Configuration

Copy `.env.example` to `.env` and override values as needed.

Scanned PDFs and images must be submitted to the approved OCR service in
`New_current/`.

Entity recognition requires SciSpaCy and the configured model, default
`en_core_sci_sm`, to be installed in the Python environment.

ModernBERT processing requires PyTorch, Transformers, and the configured
HuggingFace model, default `answerdotai/ModernBERT-base`.

Clinical context processing uses the configured BioClinicalBERT/OpenMed-compatible
encoder, default `emilyalsentzer/Bio_ClinicalBERT`.

Qwen3 simplification uses the configured Qwen3 model, default `Qwen/Qwen3-0.6B`,
and loads its prompt from `app/prompts/qwen3_simplification_prompt.txt`.

Granite Guardian validation uses the configured model, default
`ibm-granite/granite-guardian-3.0-2b`, plus deterministic checks for terms,
meanings, numbers, and unsupported care advice. Failed unsafe checks are
rejected; other factual failures recommend regeneration.

Evaluation computes BERTScore, cosine semantic similarity, Flesch-Kincaid Grade
Level, Flesch Reading Ease, and medical consistency against fused structured
facts.

# Technical Implementation Timeline — Medical Term Simplifier

Status: internship submission document
Prepared: 2026-08-10
Source of truth: repository file evidence, `IMPLEMENTATION_LOG.md`, `ROADMAP.md`, `ARCHITECTURE.md`, `MODEL_MANIFEST.md`

## How this timeline was reconstructed

The repository at `E:\Internships\Icebrkr\Medical-Term-Simplifier` has **no usable Git history** — `.git/` exists but contains no objects, refs, or commits (`git log`/`git status` both fail with "not a git repository"). No claim in this document is based on commit history, author metadata, or `git blame`; this timeline is reconstructed from repository documentation and file modification metadata only, from two evidence sources:

1. **File modification timestamps**, filtered to exclude vendored/cache directories (`.venv`, `.model-cache`, `__pycache__`, `.pytest_cache`, `.ruff_cache`, `*.egg-info`) that were bulk-touched by an environment setup/restore event on 2026-08-01 and are not meaningful development evidence. After filtering, 346 real source/doc files carry genuine, spread-out timestamps from 2026-06-22 through 2026-08-09.
2. **`IMPLEMENTATION_LOG.md`**, an append-only, self-dated engineering log maintained inside `New_current/` starting 2026-08-02, which records phase, status, verification evidence, and known limitations for every change from that date forward. This is the authoritative source for the 2026-08-02 to 2026-08-09 period and is quoted/summarized directly below.

Where a date is inferred from a file timestamp rather than stated explicitly in a log entry, it is marked "file evidence." Two calendar gaps (2026-06-29 to 2026-07-05, and 2026-07-13 to 2026-07-19) have **no corresponding file evidence** anywhere in the repository and are not filled in — per the reporting rules, no work is claimed for periods without evidence.

---

## 2026-06-22 — Evaluation framework initiated

- `Evaluation/test.py` created (file evidence). Earliest timestamped file in the repository.

## 2026-06-26 — Model comparison benchmark built (file evidence, `Evaluation/`)

- Built a benchmarking framework comparing `ibm-granite/granite-4.0-h-tiny` against `Qwen/Qwen3-4B` for three-level medical report simplification.
- Delivered: `benchmark.py` (CLI runner), `evaluation.py` (metric aggregation, weighted 0–100 readiness score), `entity_metrics.py` (precision/recall/F1, hallucination detection, critical-safety regex checks), `readability.py` (Flesch/SMOG/Gunning-Fog via `textstat`), `semantic.py` (embedding cosine similarity), `performance.py` (latency/RAM/VRAM profiling), `plots.py` (matplotlib visualizations), `generate_synthetic_dataset.py` (1,000 template-based synthetic reports across 16 specialties), `utils.py`, `config.json`.
- Result artifacts: `Evaluation/metrics/qwen_metrics.json`, `Evaluation/metrics/granite_metrics.json`, `Evaluation/comparison_report.md`, `Evaluation/reports.csv` (1,000 rows), `Evaluation/plots/*.png`.
- **Finding**: `qwen_metrics.json` records a readiness score of 44.48/100 for Qwen3-4B versus 18.79/100 for Granite in `granite_metrics.json` — both are the harness's own saved output — but the narrative in `comparison_report.md` only tabulates the Granite row and recommends Granite. The written report is internally inconsistent with the numbers its own pipeline produced. This is recorded here as a factual description of what the artifacts contain; it is **not** evidence that either model is objectively superior — both candidates scored weakly in absolute terms (e.g. entity recall of 0.0333 for Granite and 0.28 for Qwen) on a single run against a template-generated dataset, and the historical evidence is preserved rather than silently corrected.

## 2026-07-06 to 2026-07-07 — FastAPI staged pipeline backend ("Stage 11") (file evidence, `backend/`)

- Built a full FastAPI backend implementing a staged pipeline: document parsing (PyMuPDF/pdfplumber) → section segmentation → SciSpaCy entity recognition → ModernBERT difficult-term detection → BioClinicalBERT/OpenMed semantic interpretation → a hand-specified `weighted-key-match-v1` fusion algorithm → Qwen3 simplification → IBM Granite Guardian safety validation → BERTScore/readability/consistency evaluation.
- Per-stage and end-to-end REST endpoints under `backend/app/api/routes/` (e.g. `/reports/extract`, `/reports/segment`, `/reports/entities`, `/reports/fusion`, `/reports/simplify`, `/reports/validate`, `/reports/evaluate`).
- 11 test files (1,487 lines) using FastAPI `TestClient` with real assertions and dependency-injected fakes at model boundaries (e.g. `FakeQwenBackend` in `test_qwen_simplification.py`).
- **Implemented and unit-tested** (test logic verified by direct inspection). **Not verifiable** whether the suite currently executes end-to-end in this workspace — no `pytest` is installed in `backend/.venv`.

## 2026-07-11 — Standalone CLI pipeline (file evidence, `pipeline testing/`)

- Built a second, independent implementation: a local-only CLI pipeline (no web layer) chaining `TextExtractor` → `MedicalTextPreprocessor` → `OpenMedNER` → `ClinicalEmbedder` → `QwenSimplifier` → `GraniteValidator` → `ReportEvaluator`, orchestrated by `MedicalSimplifierPipeline` in `pipeline.py`.
- One real `pytest` integration-style test (`tests/test_pipeline.py`) using fakes for model-backed stages and asserting real output files are written.
- **Implemented and unit-tested.** No `outputs/` artifacts are present in the directory, so there is no evidence this pipeline was run to completion against its sample input (`reports/sample.txt`).

## 2026-07-20 to 2026-07-21 — Schema and architecture reference materials (file evidence, repository root)

- `DB.pdf` (2026-07-20) — the original logical database schema diagram, later transcribed into `ARCHITECTURE.md` §7.
- `MedicalTermSimplifierFlowchart.png` / `.html` (2026-07-21) — a rendered pipeline flowchart.

## 2026-07-26 — `New_current/` production line inaugurated (file evidence)

- `New_current/app/clinical/__init__.py` is the first file created under what would become the production source boundary.

## 2026-07-28 to 2026-07-29 — Early performance experiments in `New_current/` (file evidence + `PERFORMANCE_OPTIMIZATION.md`, `QWEN_PROFILING.md`)

- GLiNER-biomed context-window reduction experiment: `benchmarks/gliner_context_results.json`, documented in `PERFORMANCE_OPTIMIZATION.md`. Reported before/after latency reduction (e.g. 10-page report 114.99 s → 92.05 s, a 20.0% reduction) and a 77.2% reduction in evidence-text size at 10 pages.
- Qwen3-0.6B generation-parameter profiling: `benchmarks/qwen_profile_before.json` / `qwen_profile_after.json`, documented in `QWEN_PROFILING.md`. Reported generation time 40,692.8 ms → 11,418.1 ms (71.9% reduction), throughput 0.197 → 0.701 tokens/s (3.56x), at a measured RAM cost increase of 1,223.3 MB.
- **Important caveat for this pair of reports**: both documents benchmark a GLiNER-biomed + Qwen3-0.6B service composition living directly inside `app/main.py`'s lifespan. That composition is **not present in the current `New_current/app/main.py`** (verified by direct inspection — no GLiNER import exists there today; GLiNER only exists under `app/clinical/gliner_medical.py` and the separate `benchmarks/ner/` evaluation harness). These two reports should be read as **historical benchmark evidence from an earlier internal architecture**, not as a description of the current production route surface.
- `PRODUCTION_DEPLOYMENT.md` (2026-07-29) similarly describes a `/process-report`, `/reports/{job_id}/status` API and `BackgroundTasks`-based job queue design. This does not match the current implementation, which uses `/api/v1/jobs` backed by Celery, Redis, and PostgreSQL (`New_current/app/infrastructure/`). Treated as an earlier design proposal, not delivered architecture.

## 2026-07-30 to 2026-07-31 — Synthetic SFT dataset pipeline (file evidence + `sft_data_pipeline/README.md`, manifest)

- Built two parallel dataset-generation paths:
  - A live-LLM pipeline (`main.py`, `generator.py`, `validator.py`, `checkpoint.py`) that calls a configurable provider (OpenAI/OpenRouter/Claude) to generate instruction-tuning samples, with strict JSON-schema validation and crash-consistent checkpointing. **No run evidence exists** for this path (no output JSONL, checkpoint file, or log file present).
  - A deterministic, template-based generator (`generate_5000_dataset.py`) that **did** run to completion and produced the shipped corpus.
- Deliverable: `medical_simplifier_synthetic_5000.jsonl` — verified 5,000 lines, matching `medical_simplifier_synthetic_5000_manifest.json` (seed `20260731`, SHA-256 `479b192a71cb9f5209fb122da5fa5be6f47123db868c3491621999e05e4eab13`, 27,286,030 bytes). Manifest records distribution across 16 specialties, 32 named conditions, 15 report types, three difficulty tiers (Long/Medium/Short, ~1,666/1,667/1,667 each), and per-entity-type counts.
- **Implemented and locally executable** for the deterministic path; the live-LLM path is **implemented but not verifiable as executed**.

## 2026-08-01 — QLoRA fine-tuning attempt (file evidence + `medical_term_sft/README.md`, `outputs/`, `checkpoints/`)

- Built a QLoRA supervised fine-tuning pipeline targeting `Qwen/Qwen3-4B-Instruct-2507`: 4-bit NF4 double-quantization, LoRA rank 32/alpha 64 on `q_proj/k_proj/v_proj/o_proj`, assistant-only loss masking, TRL `SFTTrainer`, deterministic 80/10/10 dataset split (SHA-256 fingerprint matches the `sft_data_pipeline` corpus), checkpoint-restart support.
- Two configuration profiles: `configs/qwen3.yaml` (full-size, max sequence length 4096) and `configs/qwen3_8gb.yaml` (memory-constrained, max sequence length 2048). `outputs/resolved_config.json` confirms the actual run used the 8GB profile.
- `outputs/training_metrics.csv` records 27 logged steps, training loss falling from 1.12 to 0.097, reaching `epoch: 1.0` of a configured 3 epochs. **Validation loss was never recorded** (empty column for every row).
- `checkpoints/` contains only a TensorBoard event file — **no adapter weight files, `latest.json`/`best.json` pointers, `evaluation_metrics.json`, or `predictions.json` exist**. TensorBoard event timestamp: 2026-08-01 23:49:04.
- **Status: training started and progressed through approximately one epoch of three configured, then stopped without producing a saved adapter checkpoint or evaluation results.** This must not be described as a completed or deployable fine-tune.

## 2026-08-02 — Phase 1: Repository Infrastructure — `COMPLETE`

Source: `IMPLEMENTATION_LOG.md` (2026-08-02 entry).

- Added `AGENTS.md`, `ARCHITECTURE.md`, `ROADMAP.md`, and `IMPLEMENTATION_LOG.md` as the standing engineering contract for `New_current/`.
- Verified the logical database schema against `DB.pdf` and confirmed PostgreSQL, Redis, Celery, Qdrant, Docker, and versioned `/api/v1/` routes were not yet implemented at this point.
- Result: repository governance and phase-gating rules established; Phase 2 (OCR Service) opened as the active phase.

## 2026-08-03 — Phase 2: OCR Service, Incremental Steps 1–4 + Architecture Convergence — `IN PROGRESS`

Source: `IMPLEMENTATION_LOG.md` (five dated 2026-08-03 entries).

- **Step 1**: Established the isolated `New_current/app/ocr/` bounded context (api/application/domain/providers/postprocessing/infrastructure/observability packages) without changing existing behavior.
- **Step 2**: Added persistence-neutral `OCRRequestRecord`/`OCRResultRecord` domain models and repository/unit-of-work interface contracts (no database wired yet).
- **Step 3**: Added provider contracts, an instance-scoped registry with entry-point discovery, factories, lifecycle management, environment-backed configuration, and `GET /api/v1/ocr/health`. Model inference remained unimplemented by design (`NotImplementedError` skeletons). 70 tests passing.
- **Step 4**: Replaced skeletons with production `Qwen3-VL` OCR and `SymSpell` post-processing adapters — lazy model loading, CUDA/CPU device resolution, batching for PDF/PNG/JPEG/TIFF, conservative document decoding (decompression-bomb handling, encrypted-PDF rejection), and a three-stage post-processor (regex → medical-abbreviation dictionary → SymSpell) with idempotent abbreviation protection. 82 tests passing (1 CUDA-conditional skip). **No approved model checkpoint was cached locally yet**, so live inference remained unverified at this step.
- **Architecture Convergence**: Removed a competing legacy `ReportIngestionPipeline` (PaddleOCR/TrOCR-based) entirely, consolidating on one `OCRApplicationService` orchestration path with versioned REST APIs and a Jinja2/Bootstrap engineering console. 42 tests passing.
- **Production Validation** (same day): Ran `benchmarks/ocr/run_validation.py` against the real provider stack. Decoder checks passed for all five supported formats; post-processing measured 13 corrections in 0.472 ms. Model checkpoint was still not locally cached — CER/WER, live inference latency, and confidence-distribution metrics were explicitly recorded as `NOT VERIFIED`. Formal recommendation: **NOT READY FOR PHASE 3**.
- **Deterministic Model Validation Finalization**: Added root-level `MODEL_MANIFEST.md` as the sole source of approved model identity, with a strict fail-closed loader. Removed an unapproved external OCR executable integration entirely. 45 tests passing.

## 2026-08-04 — Phase 2 simplified; Phase 5 (Medical NER) Stages 1–2 opened

Source: `IMPLEMENTATION_LOG.md` (three dated 2026-08-04 entries).

- **Single-Model OCR Architecture Simplification**: Removed the separate document-classifier provider entirely; Qwen3-VL now infers document type and OCR text in one structured generation. The pinned `Qwen/Qwen3-VL-4B-Instruct` (revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`) loaded from local cache on CPU in 6,780.428 ms; `GET /api/v1/ocr/health` returned `ready`. 46 tests passing.
- **Phase 5 Stage 1 — Evaluation Framework**: Built a provider-neutral `app.ner` evaluation boundary registering three benchmark candidates (OpenMed Zero-Shot GLiNER, `d4data/biomedical-ner-all`, `Kushtrim/ModernBERT-base-biomedical-ner`) with exact-span P/R/F1 scoring, latency/RAM/GPU-memory measurement, and an evaluation-only `/api/v1/ner/benchmark` endpoint. 54 tests passing.
- **Phase 5 Stage 2 — Candidate Evaluation**: Ran all three candidates offline on the same synthetic 4-record, 15-entity dataset. `biomedical-ner-all` ranked first (macro F1 0.381250, mean latency 22.65 ms), ahead of ModernBERT (macro F1 0.170833) and OpenMed GLiNER (macro F1 0.083333). Recorded as a recommendation only (`winner: null` in the report); no production model selected yet.

## 2026-08-05 — Phase 5 Stage 3 production integration; Phases 6–8 boundaries; MVP slice; Phases 3–4 software boundary

Source: `IMPLEMENTATION_LOG.md` (nine dated 2026-08-05 entries — the single busiest day in the repository).

- **Phase 5 Stage 3 — Production Integration**: Formally approved `d4data/biomedical-ner-all` (revision `015a4050c9ac99722e61c547aa9b4282bcbedc7f`, Apache-2.0) as the sole production NER provider; removed benchmark endpoints from production startup. Added production `POST /api/v1/ner`. Real inference on a live sample returned HTTP 200 in 49.021 ms with measured confidence 0.872816. Corrected a live-validation defect in checkpoint-native label mapping (dropped continuation labels) and added overlapping-window chunking for inputs beyond 512 tokens. 58 tests passing.
- **Phase 6 — Entity Linking**: Built the full provider/service/API boundary for SciSpaCy + UMLS linking. Production readiness reports `not_configured` because the exact SciSpaCy version, language model, and licensed UMLS release remain unapproved. 64 tests passing.
- **Phase 7 — Relation Extraction (deferred)**: Built the provider/service/API boundary around `michiyasunaga/BioLinkBERT-base` (revision `b71f5d70f063d1c8f1124070ce86f1ee463ca1fe`). Inspected the cached checkpoint's `config.json` and found it declares a base `BertModel` with no trained relation-classification head; the provider explicitly refuses to initialize a random classification head and reports `incompatible_artifact`. This is architecture-complete, runtime-deferred by design.
- **Phase 8 — Medical Embeddings**: Built the BioClinical ModernBERT provider boundary (attention-mask mean pooling, optional L2 normalization). Production readiness reports `not_configured` because exact repository ID/revision/license remain unapproved. 70 tests passing.
- **Consolidated Engineering Demonstration Dashboard**: Added `/engineering-demo`, a same-origin Jinja2/Bootstrap page composing existing health/inference endpoints only. 72 tests passing.
- **MVP Patient Workflow Slice**: Defined and froze the MVP sequence — Upload → OCR → Medical NER → Qwen3 Simplification → IndicTrans2 Translation — with Entity Linking, Relation Extraction, Verification, and TTS explicitly marked **Deferred for MVP** (architecture preserved, not invoked). Added `POST /api/v1/simplifications` and `POST /api/v1/translations`. 74 tests passing.
- **MVP Upload Processing Fix**: Diagnosed and fixed a real performance defect — a 4-page PDF with a native text layer was being unnecessarily rendered to images and run through the 4B vision OCR model on CPU (~11.6 GB RSS, multi-minute latency). Added a bounded native-PDF text-layer fast path. 75 tests passing.
- **Phases 3 and 4 — Production Infrastructure Software Boundary**: Completed async SQLAlchemy 2.x models (17 tables), the single-head Alembic migration `0001_initial_schema`, encrypted Redis caching, Celery CPU/GPU queues with durable job state and Beat-based broker recovery, `POST/GET /api/v1/jobs`, and full Docker Compose topology (postgres, redis, migrate, api, celery-worker, celery-beat). 84 tests passing, 107 application modules import-verified. **No PostgreSQL, Redis, or Docker executable was available on the development host**, so live-service validation (real upgrade/downgrade, real Redis expiry/locking, real worker restart) remains explicitly unverified; only offline DDL generation and fake-backed adapters were exercised.

## 2026-08-06 to 2026-08-07 — Phase 9: Qwen Simplification production boundary

Source: `IMPLEMENTATION_LOG.md` (2026-08-06 entry); `SIMPLIFICATION_INTEGRATION_REPORT.md` (2026-08-07).

- Replaced the MVP single-output adapter with a three-level (Clinical / General Public / Child-Friendly) provider contract behind `POST /api/v1/simplify`. Added the external versioned prompt `qwen-medical-simplification-v2` and a deterministic source-grounding guard that fails closed when output introduces a numeric value or explanation not present in the source text.
- Real inference on a synthetic negated pneumonia/medication report produced all three levels in 91,078.880 ms with measured preservation score 1.0 for every level. A separate real inference on a synthetic HbA1c (numeric laboratory) report was **correctly rejected** because the model introduced an unsupported numeric detail — evidence the fail-closed safety guard functions under real model behavior, at the cost of the guard remaining unresolved for numeric/laboratory content. 86 tests passing.

## 2026-08-09 — OCR reliability verification, GPU enablement, Translation/Verification validation, performance experiments, observability

Source: `IMPLEMENTATION_LOG.md` (twelve dated 2026-08-09 entries — the second-busiest day in the repository).

- **OCR Reliability Verification**: Read the complete OCR module against its own tests; found and fixed a benchmark-tooling defect in `run_validation.py` where one failing document discarded already-passing evidence for its whole device run. Ran live smoke tests against the real pinned Qwen3-VL checkpoint (18.1 s init, confidence 0.9999 on a synthetic transcription) and the real pinned NER checkpoint (32.2 s init, 8 real entities extracted). No production code defect was found.
- **Cross-Phase MVP Inference Performance Optimization**: Diagnosed that every stage was running on CPU despite a physical RTX 5050 Laptop GPU (8 GB) because the installed PyTorch build had no CUDA support and both the OCR and NER providers defaulted to a hardcoded `cpu` device. Installed a CUDA-enabled PyTorch build, changed OCR/NER device defaults to `auto`, and added a real batched `translate_batch` path. Measured before/after: OCR 168.7 s → 17.2 s (9.8x), NER 1.30 s → 0.77 s (1.7x), Simplification 463.3 s → 135.4 s (3.4x) on the same host. 86 tests passing.
- **Phase 11 — IndicTrans2 Runtime Validation and Approved-Artifact E2E Validation**: Provisioned the pinned `ai4bharat/indictrans2-en-indic-dist-200M` checkpoint (revision `173b94239f7c38886b2747b8d4a5db771a7e1232`, MIT), verified its `model.safetensors` SHA-256 before load, and validated real Hindi/Tamil/Kannada translation with protected numeric/unit/date placeholders. A real, no-mock end-to-end run (OCR → NER → Simplification → Translation) completed in 22,185.073 ms total. 95 tests passing.
- **Phase 10 — Medical Verification Technical Validation**: Implemented the local-only PubMedBERT/MedNLI provider (`pritamdeka/PubMedBERT-MNLI-MedNLI`) with explicit `config.json`-driven label mapping. Real CUDA inference confirmed entailment→PASS, contradiction→BLOCKED, neutral→REVIEW behavior, and numeric/dosage/negation mismatches correctly returned BLOCKED. **License remains unverified**; production approval is explicitly pending.
- **Clinical Performance Benchmark Harness**: Added a versioned, model-neutral JSONL benchmark schema for nine document categories (lab, prescription, discharge, radiology, consultation, handwritten/scanned, table-heavy, small-text, multi-page). The populated dataset currently contains one record; representative clinical data remains pending.
- **PDF De-ID Synthetic OCR Benchmark and precision/resolution sweeps**: Registered a 50-document synthetic PHI dataset. Found the full FP32 Compose profile is **not deployable** on the 8 GB development GPU (CUDA out-of-memory). BF16 was identified as the lower-load-time candidate, but no image-resolution/precision combination completed OCR within the benchmark's time limit on Medium/Hard synthetic documents; no production default was changed as a result.
- **Internal Pipeline Test Console rewrite**: Rewrote `/engineering-demo` into a Step-by-Step and Full-Pipeline console driven only by real fetch responses (no simulated progress). 109 tests passing.
- **Production-Safe Runtime Observability**: Added `GET /api/v1/runtime/metrics` exposing real CUDA memory, CPU RSS, and per-stage warm/cold state, sourced from `torch.cuda` and `psutil` (no new heavy dependency for GPU-utilization percentage, which is explicitly reported as unavailable rather than substituted). 117 tests passing — the final recorded test count in the implementation log.

---

## Test-suite growth over the 2026-08-02 to 2026-08-09 period

| Date | Milestone | Passing tests |
|---|---|---|
| 2026-08-03 | OCR provider skeletons | 70 |
| 2026-08-03 | OCR production adapters | 82 |
| 2026-08-03 | Architecture convergence | 42 (legacy suite removed) |
| 2026-08-04 | NER evaluation framework | 54 |
| 2026-08-05 | NER production integration | 58 |
| 2026-08-05 | Entity Linking | 64 |
| 2026-08-05 | Embeddings | 70 |
| 2026-08-05 | Engineering demo | 72 |
| 2026-08-05 | MVP workflow slice | 74 |
| 2026-08-05 | Upload fix | 75 |
| 2026-08-05 | Infrastructure (Phases 3/4) | 84 |
| 2026-08-06 | Simplification | 86 |
| 2026-08-09 | IndicTrans2 validation | 95 |
| 2026-08-09 | Pipeline console rewrite | 109 |
| 2026-08-09 | Runtime observability | 117 |
| 2026-08-09 (independent live collection) | `pytest --collect-only` | 118 collected, 0 errors |

Source: dated entries in `IMPLEMENTATION_LOG.md`, cross-checked with a live `pytest --collect-only -q` run against `New_current/.venv` (118 tests, 0 collection errors).

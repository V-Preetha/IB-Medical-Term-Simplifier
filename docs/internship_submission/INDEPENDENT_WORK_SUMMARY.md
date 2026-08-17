# Independent Work Summary — Medical Term Simplifier

Prepared: 2026-08-10
Scope: work found in the repository at `E:\Internships\Icebrkr\Medical-Term-Simplifier`, covering the period 2026-06-22 to 2026-08-09 (reconstructed from repository documentation and file modification metadata, not from Git commit history — this repository's `.git/` directory contains no commits or author records).

**A note on attribution**: because there is no commit history or per-change author metadata in this repository, no statement below is a claim of exclusive or sole authorship verified by version-control evidence. The contributions described here are the work products found in the intern's assigned area of the repository, described using "implemented," "contributed to," "worked on," and "evaluated" rather than language that implies sole design credit for the whole system (e.g. "designed the entire system" or "built the complete architecture"). Where a specific function, fix, or adapter was traced directly to its code, that is stated plainly as an implementation contribution; broader architectural framing is described as something the work participated in and followed, not something claimed as solely originated.

## 1. Project Overview

The Medical Term Simplifier is a component of the ICEBRKR / IB Health ecosystem that converts complex medical documents — laboratory reports, discharge summaries, prescriptions, radiology reports, and consultation notes — into medically faithful, patient-understandable explanations at three readability levels (Clinical, General Public, Child-Friendly), with optional translation into Indian regional languages. The production-oriented implementation lives under `New_current/`; several earlier, superseded implementations and supporting research artifacts exist elsewhere in the repository and are documented separately in this submission package.

## 2. Independent Responsibilities

Based on repository evidence, work spanned the following areas:

- **Architecture and technical research**: contributed to `ARCHITECTURE.md`, `ROADMAP.md`, `AGENTS.md`, and `MODEL_MANIFEST.md`, the repository's standing engineering contract; built and ran a comparative LLM evaluation (Qwen vs. Granite) before a simplification base model was committed to.
- **OCR**: implemented the Qwen3-VL-based OCR service within `New_current/app/ocr/`, its post-processing pipeline (regex, medical-abbreviation dictionary, SymSpell), and the fail-closed model-manifest mechanism it uses.
- **Medical NLP (NER, entity linking, relation extraction)**: built and ran the three-candidate NER benchmark, integrated the selected production winner, and implemented the architecture boundaries for entity linking and relation extraction (both intentionally left runtime-inactive pending licensed/fine-tuned artifacts).
- **Simplification**: implemented the three-level Qwen3 simplification service and its deterministic source-grounding safety guard.
- **Fine-tuning**: developed the synthetic instruction-tuning dataset pipeline and the QLoRA fine-tuning pipeline for Qwen3.
- **Medical embeddings**: implemented the BioClinical ModernBERT embedding provider boundary (model identity not yet approved; no retrieval layer built).
- **Backend/API and database work**: implemented FastAPI service boundaries for every stage and the 17-table async SQLAlchemy schema with its Alembic migration, Redis caching, and Celery job infrastructure.
- **GPU/performance engineering**: diagnosed and fixed a CPU-only inference bottleneck across the pipeline, then enabled and measured real GPU acceleration.
- **Translation**: implemented and validated the IndicTrans2 translation service end-to-end.
- **Verification**: implemented and technically validated a PubMedBERT/MedNLI verification service (license and production approval pending).
- **Testing and documentation**: wrote test coverage accompanying each module above and maintained the append-only `IMPLEMENTATION_LOG.md`.

Areas explicitly **not** claimed here because repository evidence does not support them: text-to-speech (architecture referenced only, `DEFERRED FOR MVP`, no implementation); Qdrant vector retrieval (not implemented); a completed, deployable QLoRA fine-tune (training was initiated and its early progress was logged, but it did not finish and no adapter was saved — see Section 3).

## 3. Technical Contributions

### Evaluation of candidate simplification models (`Evaluation/`)

**Problem**: Before any model could be wired into the pipeline, there needed to be a repeatable, quantified way to compare candidates rather than relying on informal inspection — especially given the safety stakes of medical text simplification.

**Approach**: Build an evaluation harness that runs each candidate through the same prompt and scores it on six independent dimensions (semantic similarity, entity precision/recall, readability-band compliance, hallucination rate, critical-safety regex checks, and resource/latency profiling), combined into a single weighted readiness score.

**Implementation**: Built the harness (`benchmark.py`, `evaluation.py`, `entity_metrics.py`, `readability.py`, `semantic.py`, `performance.py`) and a 1,000-report deterministic synthetic dataset, and ran it against `ibm-granite/granite-4.0-h-tiny` and `Qwen/Qwen3-4B`.

**Outcome**: The run produced two separate metrics files — `granite_metrics.json` (readiness score 18.79/100) and `qwen_metrics.json` (readiness score 44.48/100) — but the generated `comparison_report.md` only tabulates the Granite row and recommends Granite, which is inconsistent with the harness's own saved output for Qwen. This documentation reports both measured scores factually and flags the report-generation inconsistency; it does not take a position on which candidate was actually superior, since both showed weak absolute scores (e.g. entity recall of 0.0333 for Granite and 0.28 for Qwen) on a single two-candidate run against a template-generated dataset — not a sufficient basis for a confident model-quality conclusion either way.

**Evidence**: `Evaluation/**`, `Evaluation/metrics/qwen_metrics.json`, `Evaluation/metrics/granite_metrics.json`, `Evaluation/comparison_report.md`.

### OCR service (`New_current/app/ocr/`)

**Problem**: Convert uploaded medical documents (PDF, PNG, JPEG, TIFF) into accurate machine-readable text while handling digital, scanned, and mixed documents differently, and never silently proceeding on low-confidence or corrupted input.

**Approach**: A layered, provider-neutral structure (api → application service → provider interface → model adapter) within the OCR module, with a native-PDF text-layer fast path ahead of a Qwen3-VL vision-model fallback, followed by a three-stage deterministic post-processor.

**Implementation**: Implemented the OCR bounded context across four incremental, independently-verified steps; participated in removing a competing legacy PaddleOCR/TrOCR pipeline during an architecture-convergence pass; implemented conservative document decoding (decompression-bomb handling, encrypted-PDF rejection, EXIF correction); implemented the `Qwen3VLOCRProvider` with lazy model loading, CUDA/CPU auto-resolution, and per-request GPU memory tracking; implemented idempotent medical-abbreviation-protected post-processing.

**Outcome**: A real, live-model-validated OCR pipeline (Qwen3-VL confidence 0.9999 on a synthetic transcription test) with a measured 9.8x GPU speedup over the CPU baseline (168.7 s → 17.2 s). Representative clinical-corpus accuracy benchmarking remains open.

**Evidence**: `New_current/app/ocr/**`, `New_current/MODEL_MANIFEST.md` (Qwen3-VL entry), `IMPLEMENTATION_LOG.md` (2026-08-03/04/09 OCR entries), `New_current/OCR_PHASE_COMPLETION_REPORT.md`, `New_current/ARCHITECTURE_CONVERGENCE_REPORT.md`, `New_current/tests/ocr/**` (10 files).

### Medical NER benchmarking and production integration (`New_current/app/ner/`)

**Problem**: Select a medical named-entity-recognition model with defensible, reproducible evidence rather than an informal choice, then integrate it safely.

**Approach**: Build an evaluation harness registering three real candidates, score them on the same synthetic corpus using exact-span precision/recall/F1 plus resource metrics, then integrate only the reviewed winner behind a provider interface that other code never bypasses.

**Implementation**: Benchmarked OpenMed Zero-Shot GLiNER, `d4data/biomedical-ner-all`, and `Kushtrim/ModernBERT-base-biomedical-ner`; `biomedical-ner-all` scored highest on macro F1 (0.381250 vs. 0.170833 and 0.083333). Integrated it as the sole production provider and removed benchmark code from production startup entirely. Found and fixed a real defect during live validation — the checkpoint-native label mapping was dropping continuation labels — and added overlapping-window chunking for inputs longer than the model's 512-token limit.

**Outcome**: Production `POST /api/v1/ner` with real-inference-verified latency (49.021 ms) and measured confidence (0.872816) on a live sample; a 669-token input crossing the chunking boundary was verified to return correctly offset spans.

**Evidence**: `New_current/app/ner/**`, `New_current/benchmarks/ner/reports/*` (JSON/CSV/Markdown), `New_current/MODEL_MANIFEST.md` (NER section), `New_current/NER_PRODUCTION_INTEGRATION_REPORT.md`, `IMPLEMENTATION_LOG.md` (2026-08-04/05 NER entries).

### Qwen3 simplification service with source-grounding safety guard (`New_current/app/simplification/`)

**Problem**: Generate three-level (Clinical/General Public/Child-Friendly) patient-friendly explanations from a single local model call, without allowing the model to introduce clinical facts not present in the source document.

**Approach**: A single deterministic generation producing all three levels, governed by an externally versioned prompt and a post-generation grounding check that fails closed on any unsupported numeric value or unexplained-but-invented term.

**Implementation**: Implemented `POST /api/v1/simplify`, the `qwen-medical-simplification-v2` external prompt, and the grounding guard. Verified the guard against two real cases: a negated pneumonia/medication report (all three levels passed with preservation score 1.0) and an HbA1c laboratory report, which was **correctly rejected** when the model introduced an unsupported number — direct evidence the guard functions under genuine model failure, not just in a designed pass case.

**Outcome**: A working, safety-gated simplification service; representative clinical faithfulness/readability benchmarking across report types remains an open roadmap item, particularly for numeric/laboratory content.

**Evidence**: `New_current/app/simplification/**`, `New_current/app/simplification/prompts/medical_report_v2.json`, `New_current/SIMPLIFICATION_INTEGRATION_REPORT.md`, `IMPLEMENTATION_LOG.md` (2026-08-06/07 entries).

### GPU inference-performance diagnosis and enablement (cross-cutting, 2026-08-09)

**Problem**: The MVP pipeline was running entirely on CPU despite a physical 8 GB CUDA-capable GPU being present on the development host, producing multi-minute latency per request.

**Approach**: Root-cause the discrepancy rather than assuming a hardware limitation.

**Implementation**: Diagnosed two independent causes — the installed PyTorch build had no CUDA support at all, and the OCR and NER providers separately defaulted to a hardcoded `cpu` device with no `auto` mode. Installed a CUDA-enabled PyTorch build, verified it with a real matmul kernel, changed both providers to `auto` device resolution with GPU dtype selection (BF16-preferred), added a real batched translation path, and fixed a frontend defect where embeddings was blocking simplification from starting (contrary to the documented policy that embeddings must stay non-blocking).

**Outcome**: Measured, reproducible speedups on the same host: OCR 168.7 s → 17.2 s (9.8x), NER 1.30 s → 0.77 s (1.7x), Simplification 463.3 s → 135.4 s (3.4x). Investigated and documented the root cause of the smaller NER/Simplification speedups (Windows WDDM driver launch-latency dominating small-model wall-clock time), rather than leaving it unexplained.

**Evidence**: `IMPLEMENTATION_LOG.md` (2026-08-09, "Cross-Phase MVP Inference Performance Optimization"), `PERFORMANCE_PROFILE.md`, `QWEN_TARGETED_PERFORMANCE_REPORT.md`.

### Translation service with checksum-verified model loading (`New_current/app/translation/`)

**Problem**: Translate simplified reports into Indian regional languages while exactly preserving protected clinical values (dosages, units, dates) and never loading a tampered or wrong model artifact.

**Approach**: Local-only IndicTrans2 inference with a bracketed-sentinel value-protection scheme and mandatory pre-load SHA-256 verification of the exact weight file.

**Implementation**: Provisioned the pinned checkpoint, fixed two real defects found during real inference (an incompatible Transformers KV-cache path for this custom checkpoint, and a sentinel scheme that did not survive Indic-script transliteration), and added batched multi-text translation.

**Outcome**: Verified real translation into Hindi, Tamil, and Kannada with exact preservation of tested dosage/unit/date values, plus a real, no-mock, five-stage end-to-end pipeline run (OCR → NER → Simplification → Translation) completing in 22,185.073 ms.

**Evidence**: `New_current/app/translation/**`, `New_current/benchmarks/translation/artifacts/2026-08-09-approved/`, `TRANSLATION_E2E_VALIDATION_REPORT.md`, `IMPLEMENTATION_LOG.md` (2026-08-09 Phase 11 entries).

### Production infrastructure software boundary (`New_current/app/db/`, `app/infrastructure/`)

**Problem**: Provide durable, tenant-scoped storage and background-job processing matching the authoritative `DB.pdf` schema, without a live PostgreSQL/Redis/Docker environment available for testing.

**Approach**: Build the software layer (models, repositories, migration, cache adapter, task queue, APIs) against protocol interfaces that can be exercised with fakes/SQLite locally, and record explicitly that live-service behavior remains unverified rather than assuming it works.

**Implementation**: Implemented 17 async SQLAlchemy models (11 from the original schema, 6 additive), the single-head Alembic migration, an encrypted Redis cache adapter with versioned keys, Celery CPU/GPU queues with retry/recovery logic, `POST/GET /api/v1/jobs`, and the Docker Compose topology.

**Outcome**: A software-complete, contract-tested infrastructure layer (84 passing tests at this milestone) with an explicitly documented gap: no PostgreSQL, Redis, or Docker executable exists on the development host, so live migration/cache/queue behavior is recorded as `NOT VERIFIED` rather than assumed to work.

**Evidence**: `New_current/app/db/models.py`, `New_current/migrations/versions/0001_initial_schema.py`, `New_current/app/infrastructure/**`, `New_current/docker-compose.yml`, `New_current/INFRASTRUCTURE_IMPLEMENTATION_REPORT.md`.

### Synthetic instruction-tuning dataset and QLoRA fine-tuning pipeline (`sft_data_pipeline/`, `medical_term_sft/`)

**Problem**: Produce training data and a fine-tuning pipeline for a smaller, specialized Qwen3 simplification model.

**Approach**: Build a deterministic, self-auditing synthetic-data generator (rather than depending on a live third-party LLM API for the primary corpus), and a QLoRA pipeline sized for an 8 GB development GPU.

**Implementation**: Developed the Qwen3 QLoRA/SFT training pipeline (4-bit NF4 quantization, LoRA rank 32, assistant-only loss masking, checkpoint-restart, two configuration profiles) and the synthetic medical simplification dataset that feeds it — generated and independently re-validated a 5,000-example corpus (SHA-256-fingerprinted, spanning 16 specialties and three difficulty tiers).

**Outcome**: The dataset generation is complete and verified. Training was initiated and its progress was logged — training loss fell from 1.12 to 0.097 across 27 logged steps, reaching approximately one of three configured epochs — but **the final adapter training was not completed and no production adapter weights were delivered during the documented period**. Validation loss was never recorded and no adapter checkpoint was saved. This must be described as an interrupted training run, not a delivered fine-tuned model.

**Evidence**: `sft_data_pipeline/medical_simplifier_synthetic_5000_manifest.json`, `sft_data_pipeline/medical_simplifier_synthetic_5000.jsonl` (verified 5,000 lines), `medical_term_sft/outputs/training_metrics.csv`, `medical_term_sft/checkpoints/` (contains only a TensorBoard event file, no adapter weights).

## 4. Technical Decisions Reflected in This Work

The decisions below are recorded because they are directly evidenced in the code and configuration this work touched; where a decision applies repository-wide, it is described as a standard this work followed and applied consistently, not as something originated in isolation.

- **Model selection by benchmark, not preference**: both the Qwen-vs-Granite comparison and the NER three-candidate benchmark were run and reviewed before any production integration decision.
- **Local-first, Apache-2.0-preferring model stack**: Qwen3-VL, `d4data/biomedical-ner-all`, Qwen3-0.6B, and BioLinkBERT are all Apache-2.0 licensed and loaded local-only with `local_files_only=True`; IndicTrans2 is MIT-licensed. PubMedBERT/MedNLI's license is explicitly left unresolved rather than assumed permissive.
- **Fail-closed model loading**: every stage requires an exact repository ID and immutable revision recorded in `MODEL_MANIFEST.md` or an approved environment variable before it will initialize; `PENDING_APPROVAL` is treated as a hard block, never a default. This pattern was applied consistently to each stage implemented in this work (OCR, NER, Simplification, Translation, Verification).
- **Modular, replaceable provider structure**: every AI stage implemented in this work follows the same `route → application service → provider interface → adapter` dependency direction, so no route or service imports a model library directly.
- **CUDA/CPU fallback with explicit failure on mismatch**: `auto` device resolution selects CUDA when available and otherwise CPU; an explicit `cuda` request on a CPU-only host fails loudly rather than silently downgrading — applied when fixing the OCR and NER device defaults during the GPU-enablement work.
- **Simplify-before-translate pipeline order**: reduces the burden on the translation model by translating already-simplified, shorter text rather than the full source report.
- **Native-PDF fast path before OCR**: avoids unnecessary and expensive vision-model inference on documents that already contain a usable digital text layer.
- **Deferring Entity Linking, Relation Extraction, Verification, and TTS from the MVP path**: their architecture and APIs remain fully built and registered, but the MVP demo does not call them, because each has an unresolved prerequisite (UMLS license, fine-tuned relation-classification checkpoint, verification license, and no TTS implementation at all, respectively).

## 5. Problems Solved

- **CPU-only inference bottleneck** across the entire pipeline, root-caused to a missing CUDA-enabled PyTorch build plus two hardcoded device defaults, fixed with a measured 9.8x OCR speedup.
- **Unnecessary OCR of digitally-native PDFs**, discovered only once the MVP flow was exercised end-to-end (~11.6 GB RSS, multi-minute latency), fixed with a native-text-layer fast path.
- **NER label-mapping defect** (dropped continuation labels) and **512-token truncation** on long inputs, both discovered through real-checkpoint inference rather than assumed from unit tests alone, and both fixed with regression coverage.
- **IndicTrans2 KV-cache incompatibility** and a **transliteration-fragile value-protection scheme**, both discovered through real multilingual inference and fixed before the checkpoint was approved for MVP use.
- **A benchmark-tooling defect** in the OCR validation runner that silently discarded already-passing evidence when one document failed, corrected so future runs report per-format results honestly instead of an opaque overall failure.
- **A relation-extraction checkpoint integrity gap**: the cached `BioLinkBERT-base` checkpoint has no trained classification head; rather than letting Transformers silently initialize a random head (which would produce fabricated, clinically unsafe relation output), the provider explicitly detects and rejects this state.

## 6. Deliverables

| Deliverable | Description | Status | Evidence |
|---|---|---|---|
| Model evaluation harness | 6-metric comparison of Qwen3-4B vs. Granite-4.0-h-tiny | Implemented, executed; report-generation inconsistency documented | `Evaluation/**` |
| OCR service | Qwen3-VL provider + regex/dictionary/SymSpell post-processing, versioned REST API | Implemented, model-validated; clinical corpus benchmark open | `New_current/app/ocr/**`, `IMPLEMENTATION_LOG.md` 2026-08-09 |
| Medical NER service | `biomedical-ner-all` provider, versioned REST API, benchmark evidence for 3 candidates | Implemented, model-validated; clinical corpus benchmark open | `New_current/app/ner/**`, `New_current/benchmarks/ner/reports/` |
| Entity Linking boundary | SciSpaCy + UMLS provider architecture | Architecture complete; runtime not configured (UMLS license pending) | `New_current/app/entity_linking/**` |
| Relation Extraction boundary | BioLinkBERT provider architecture with checkpoint-integrity guard | Architecture complete; runtime deferred (no fine-tuned checkpoint) | `New_current/app/relation_extraction/**` |
| Medical Embeddings boundary | BioClinical ModernBERT provider | Architecture complete; model identity unapproved; no retrieval layer | `New_current/app/embeddings/**` |
| Simplification service | Qwen3-0.6B, 3-level output, source-grounding guard | Implemented, model-validated with a real rejection case | `New_current/app/simplification/**` |
| Translation service | IndicTrans2, checksum-verified loading, batch translation | Implemented, model-validated across 3 languages + real E2E run | `New_current/app/translation/**`, `TRANSLATION_E2E_VALIDATION_REPORT.md` |
| Verification service | PubMedBERT/MedNLI, deterministic grounding checks | Technically verified; license/production approval pending | `New_current/app/verification/**` |
| Database layer | 17-table async SQLAlchemy schema + Alembic migration | Software implemented; live PostgreSQL validation pending | `New_current/app/db/**`, `New_current/migrations/versions/0001_initial_schema.py` |
| Redis/Celery infrastructure | Encrypted cache, durable job queues, jobs API, Docker Compose | Software implemented; live service validation pending | `New_current/app/infrastructure/**`, `New_current/docker-compose.yml` |
| GPU performance optimization | CUDA enablement across OCR/NER/Simplification/Translation | Implemented and measured | `IMPLEMENTATION_LOG.md` 2026-08-09, `PERFORMANCE_PROFILE.md` |
| Runtime observability endpoint | `GET /api/v1/runtime/metrics` | Implemented and tested | `New_current/app/infrastructure/routes.py` |
| Synthetic SFT dataset | 5,000-example, SHA-256-fingerprinted instruction dataset | Completed and verified | `sft_data_pipeline/medical_simplifier_synthetic_5000.jsonl` |
| QLoRA fine-tuning pipeline | Training/eval/inference scripts for a Qwen3-4B LoRA adapter | Pipeline developed; training incomplete; no final adapter delivered | `medical_term_sft/**` |
| Clinical performance benchmark harness | Offline, 9-category benchmark harness | Implemented; representative dataset not yet populated | `New_current/benchmarks/clinical_performance/**` |
| Engineering documentation | `ARCHITECTURE.md`, `ROADMAP.md`, `AGENTS.md`, `MODEL_MANIFEST.md`, `IMPLEMENTATION_LOG.md` | Completed, actively maintained | repository root |
| Internal engineering console | `/engineering-demo` Jinja2/Bootstrap dashboard | Implemented and tested | `New_current/app/templates/engineering_demo.html` |

## 7. Skills / Technologies Applied

Python, FastAPI, Pydantic, async SQLAlchemy 2.x, Alembic, PostgreSQL (schema/design), Redis, Celery, Docker/Docker Compose, PyTorch (CUDA/CPU device management, BF16/FP16 precision, `torch.cuda` memory instrumentation), Hugging Face Transformers, PEFT/LoRA, TRL (`SFTTrainer`), `bitsandbytes` (4-bit quantization), Qwen3 and Qwen3-VL model families, `d4data/biomedical-ner-all`, SciSpaCy, BioLinkBERT, BioClinical ModernBERT, IndicTrans2/IndicTransToolkit, PubMedBERT/MedNLI, SymSpell, pytest, Ruff, Jinja2/Bootstrap, OpenAPI/Swagger, structured/privacy-safe logging design, model-manifest/checksum-based supply-chain integrity practices.

## 8. Overall Contribution

Across roughly seven weeks of recorded evidence, this work contributed to a modular, provider-based backend covering the core pipeline — OCR, medical NER, simplification, translation, and verification — with each stage independently testable, health-checked, and gated behind an explicit, fail-closed model-approval manifest. Five of those stages (OCR, NER, Simplification, Translation, Verification) reached real, non-mocked inference against their pinned production checkpoints by the end of the recorded period, with measured latency, confidence, and correctness evidence, including at least one genuine safety-guard rejection case for Simplification and one for Verification. GPU acceleration was diagnosed and enabled from a cold CPU-only baseline, producing a measured 9.8x OCR speedup. Database, cache, and job-queue infrastructure reached a complete, tested software layer, with the honest limitation that live PostgreSQL/Redis/Docker validation could not be performed on the available development host. Entity Linking, Relation Extraction, and text-to-speech were deliberately scoped out of the MVP rather than shipped as unvalidated stubs, each with a documented, specific reason (licensing, missing fine-tuned checkpoint, and no implementation respectively).

Two lower-confidence areas are disclosed rather than omitted: the Qwen3 QLoRA/SFT training pipeline and its synthetic dataset were developed and training was initiated, but the final adapter training did not complete and no production adapter weights were delivered in the documented period; and an early Qwen-vs-Granite evaluation report's written recommendation does not match the same evaluation run's own saved metrics — reported factually here without endorsing either model as conclusively superior. This is not described as a production-ready or fully deployed system — clinical validation, live infrastructure validation, and several licensing approvals remain explicitly open per the project's own roadmap.

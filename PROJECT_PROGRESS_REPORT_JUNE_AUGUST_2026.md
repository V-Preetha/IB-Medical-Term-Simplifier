# Medical Term Simplifier — Project Progress Report

**Project:** ICEBRKR / IB Health – Medical Term Simplifier
**Reporting Period:** June 2026 – August 17, 2026
**Status:** Active Development

---

## 1. Executive Summary

The Medical Term Simplifier converts medical documents — lab reports, discharge summaries,
prescriptions, and scanned/handwritten reports — into medically faithful, patient-friendly
explanations at three reading levels, with optional translation into Indian regional
languages. The project began in June 2026 as an architecture and design exercise and moved
into active implementation in July, with the heaviest engineering work concentrated in the
first two weeks of August.

**What has been built:** a modular FastAPI backend (`New_current/`) with independently
versioned services for OCR, Medical NER, Entity Linking, Relation Extraction, Embeddings,
Simplification, Verification, and Translation, each behind a replaceable provider interface
with its own health checks, tests, and same-origin engineering dashboard. A real, no-mock,
end-to-end run — upload → OCR → NER → Qwen3 simplification → IndicTrans2 translation — has
been executed successfully on this machine using pinned local model checkpoints.

**Major architectural changes:** the original June design assumed classical OCR (PaddleOCR/
TrOCR) and a broader always-on pipeline including entity linking, relation extraction, and
speech (originally "Project Vaani"). By August, OCR converged onto a single vision-language
model (Qwen3-VL), the MVP flow was explicitly narrowed to
`OCR → NER → Simplification → Translation` with Entity Linking, Relation Extraction,
Verification, and TTS deliberately deferred, and a mid-August GPU enablement effort moved
inference from CPU-only to CUDA, cutting single-stage latencies by roughly 2–10x.

**What currently works today:** the core MVP chain runs end-to-end with real local models
under a documented safety gate — the simplifier is designed to fail closed rather than
invent unsupported medical facts, and it has been observed doing so correctly on a synthetic
lab report. Translation preserves protected numeric/unit/dosage values across Hindi, Tamil,
and Kannada. NER, embeddings, and the OCR text pipeline are individually validated.

**What is still pending:** clinical-quality validation (no representative, clinician-reviewed
corpus exists yet), PostgreSQL/Redis/Celery are fully coded but never exercised against live
services, Entity Linking and Relation Extraction lack approved runnable model checkpoints,
Verification is technically working but not license-cleared, and TTS has no implementation.
A PP-OCRv6 candidate model was correctly rejected in mid-August due to contradictory identity
metadata rather than being silently substituted.

**Immediate next priorities:** freeze and clinically validate the OCR/NER/Simplification
chain against real de-identified reports, stand up and validate PostgreSQL/Redis/Celery
against live services, and resolve the two outstanding hygiene issues (a leaked Hugging Face
token, and an OCR artifact-cache directory that should never be committed) before any public
GitHub publication.

---

## 2. Current System Objective

The system takes a medical document — a PDF, scanned image, discharge summary, prescription,
lab report, or (in principle) a handwritten note — and produces a medically faithful
explanation a patient can actually understand, without dropping or altering clinical facts.

Target workflow:

```text
Medical Document
        ↓
Document Processing / Classification
        ↓
OCR or Vision-Language Extraction
        ↓
Text Cleaning
        ↓
Medical NER
        ↓
Entity Linking / Clinical Context
        ↓
Medical Knowledge Retrieval
        ↓
Medical Simplification
        ↓
Medical Verification
        ↓
Three Readability Levels
        ↓
Translation
        ↓
Speech Output
```

Status of each stage against this target, as of August 17, 2026:

| Stage | Status |
|---|---|
| Document Processing / OCR (Qwen3-VL) | **Implemented, runtime-validated on CPU and GPU** — clinical accuracy benchmark still pending |
| Text Cleaning (regex + abbreviation dictionary + SymSpell) | **Implemented and tested** |
| Medical NER (biomedical-ner-all) | **Implemented, runtime-validated** — clinical corpus threshold validation pending |
| Entity Linking (SciSpaCy + UMLS) | **Planned / architecture only** — no approved model, UMLS license not secured |
| Medical Knowledge Retrieval (Qdrant) | **Planned** — embeddings provider exists standalone, not wired to any retrieval store |
| Medical Simplification (Qwen3-0.6B) | **Implemented, runtime-validated** — clinical faithfulness/readability thresholds not yet set or met |
| Medical Verification (PubMedBERT MedNLI) | **Technically working, license pending** — not production-approved |
| Three Readability Levels | **Implemented** (Clinical / General Public / Child-Friendly) |
| Translation (IndicTrans2) | **Implemented, runtime-validated end-to-end** — clinical/native-speaker review pending |
| Speech Output (TTS) | **Deferred for MVP — not implemented** |

---

## 3. DEVELOPMENT TIMELINE

| Period | Focus | Work Completed | Result / Status |
|---|---|---|---|
| June 2026 | Architecture & design | Defined overall workflow, evaluated candidate models per stage, drafted DB schema and integration plan with IB Health/AIPA | Design-only; no production code yet |
| July 2026 | Early implementation & evaluation | OCR API scaffolding, document decoding, NER candidate evaluation, Qwen simplification prototype, SFT dataset generation | Working prototypes and benchmarks, not yet production-hardened |
| Aug 1–8, 2026 | Core service build-out | OCR/NER/Entity Linking/Relation Extraction/Embeddings production boundaries, database + Redis/Celery software layer, MVP pipeline slice | Individually testable services; MVP chain assembled but not GPU-accelerated |
| Aug 9, 2026 | Architecture convergence & performance | GPU/CUDA enablement, IndicTrans2 provisioning, real end-to-end run, verification technical validation, clinical benchmark harness | First real, no-mock E2E pipeline run completed |
| Aug 10–17, 2026 | Documentation & safety gating | Internship documentation set, runtime observability metrics, PP-OCRv6 artifact rejected on identity ambiguity | Repository well-documented; one candidate model correctly blocked, not shipped |

### June 2026

June was the architecture and design phase — no production code existed yet. The team
defined the intended Medical Term Simplifier workflow end to end: document ingestion, OCR,
medical NER, embeddings, simplification into three readability levels, translation, speech
output, database architecture, and integration points with IB Health/AIPA.

Early component ideas under consideration at this stage included PaddleOCR/TrOCR-style OCR,
OpenMed NER, BioClinical ModernBERT embeddings, Qwen for simplification, "Project Vaani" for
speech, PostgreSQL, Qdrant, FastAPI, Redis, and background job processing. This was the
**initial architecture**, used to scope the problem — not the final implementation, and
several of these choices were later revised (see August section).

### July 2026

Work moved from architecture into implementation and evaluation.

**OCR Pipeline.** An OCR API, document decoding, provider abstraction, and post-processing
layer were built. Supported document formats: PDF, PNG, JPEG, TIFF. Post-processing used
SymSpell, medical abbreviation dictionaries, and regex-based cleanup, exposed via
`POST /api/v1/ocr`. Classical OCR options (PaddleOCR, TrOCR) were evaluated in this baseline
but the architecture later moved toward a vision-language approach (see August).

**Handwritten / Difficult Documents.** The architecture began evolving toward a
vision-language model — the Qwen3-VL family — to handle handwriting, difficult scans, and
complex layouts. This direction is implemented and runtime-validated on CPU and GPU as of
August, but full production deployment (representative clinical benchmark, CER/WER
measurement) is not yet complete.

**Medical NER.** Candidates investigated: OpenMed GLiNER, ModernBERT biomedical NER, d4data
biomedical NER, and general GLiNER. NER identifies diseases, drugs, anatomy, symptoms, lab
tests, procedures, and other medical entities. Development profiling observed GLiNER CPU
one-page processing at roughly 12 seconds and ten-page processing at roughly 92 seconds —
these are **development profiling observations on this machine, not universal production
benchmarks**, and GLiNER was ultimately not selected for production (see August).

**Simplification Engine.** Qwen3-0.6B was set as the local simplification model direction,
producing three readability levels — Clinical, General Public, and Child-Friendly — behind an
API/service/dashboard path.

**Fine-Tuning Dataset / LoRA.** A synthetic instruction-tuning dataset of approximately 5,000
JSONL examples was generated for Qwen, using varied instruction phrasings (e.g. "Simplify this
medical report," "Explain this report to a patient," "Rewrite this report in simple English,"
"Explain this report in three readability levels," "Convert this report into patient-friendly
language," "Simplify this report without changing any medical facts") rather than one fixed
prompt.

It is important to be precise about what this dataset work represents: **dataset generation**
and a **training pipeline implementation** exist at `medical_term_sft/`. A QLoRA fine-tuning
**run was started but did not finish** — the recorded training metrics stop at epoch 1.0 of 3
configured, validation loss was never logged, and no adapter weights, `latest.json`, or
evaluation results exist in `checkpoints/`. **No trained adapter exists, and nothing from this
pipeline has been evaluated or deployed.** The production simplification path uses the base
Qwen3-0.6B model, not an output of this fine-tuning effort.

### August 2026

This is the most active period and contains the majority of production-quality work.

#### Architecture Convergence

Several June/July ideas were reconsidered as the system matured:

- **OCR / Document Understanding.** The primary direction converged on Qwen3-VL-family
  vision-language processing. PaddleOCR and TrOCR code, dependencies, and the competing
  ingestion pipeline that used them were **removed** from the production path on 2026-08-03;
  they should be read as evaluated/legacy, not current architecture.
- **"Project Vaani."** This was an earlier concept considered for runtime speech generation.
  It was **explicitly replaced** by a Kokoro TTS direction. No Vaani runtime integration
  exists anywhere in the repository — it is documented history only, and should not be
  confused with the (also unimplemented) Kokoro TTS target.

#### GPU Enablement

A significant infrastructure gap was found and fixed. The development laptop has an NVIDIA
RTX 5050 Laptop GPU with 8 GB VRAM, but PyTorch had been installed as a CPU-only build
(`2.13.0+cpu`), so `torch.cuda.is_available()` returned `False` despite the physical GPU being
present.

On 2026-08-09, `torch==2.11.0+cu128` was installed and verified with a real CUDA matmul
kernel: `torch.cuda.is_available()` returned `True` on the "NVIDIA GeForce RTX 5050 Laptop
GPU" (compute capability 12.0 / Blackwell, bf16-capable). The OCR and NER providers' device
selection was changed from a hardcoded `cpu` default to `auto`, allowing the runtime to
resolve an appropriate device instead of being pinned to CPU. This was installed into the
local development virtual environment for measurement purposes; `requirements.txt` and
`pyproject.toml` still resolve to the CPU wheel by default — the CUDA build was **not** pinned
repository-wide, since that is a deployment/hardware decision requiring separate approval
(CI and CPU-only hosts must still be able to install and run).

Measured effect on single-input smoke tests (same host, before → after):

| Stage | CPU | GPU | Speedup |
|---|---|---|---|
| OCR (Qwen3-VL-4B, synthetic PNG) | 168.7 s | 17.2 s | 9.8x |
| NER (biomedical-ner-all, 1 paragraph) | 1.30 s | 0.77 s | 1.7x |
| Simplification (Qwen3-0.6B, 3 levels, isolated) | 463.3 s | 135.4 s | 3.4x |

#### Qwen Performance Optimization

A controlled, isolated Qwen-only generation benchmark improved substantially over the course
of the GPU work:

Before:
```text
Generation ≈ 40.69 seconds
Throughput ≈ 0.197 tokens/s
Peak RAM ≈ 1.6 GB
```

After optimization:
```text
Generation ≈ 11.42 seconds
Throughput ≈ 0.701 tokens/s
Peak RAM ≈ 2.8 GB
```

Approximate controlled generation-time reduction: **71.9%**.

This controlled, isolated benchmark is **not directly comparable** to the ~91-second and later
~22-second figures reported for full pipeline runs elsewhere in this document — those used a
different prompt/fixture and measure a different thing (full multi-stage pipeline latency vs.
isolated single-model generation). Both figures are real and both are preserved, but they
answer different questions.

#### Batch Simplification

The simplification provider and its service layer were updated to support real batch
processing rather than only single-input calls. This reduces duplicated inference setup per
request, improves throughput potential, and supports better backend scalability as request
volume grows — relevant for future multi-report or multi-language batch use.

#### OCR Model Artifact Validation

A PP-OCRv6 Medium candidate was evaluated as a possible additional OCR option and was
**correctly rejected** by the system's own validation gate, which is worth highlighting as
both a limitation and a demonstration that the safety machinery works as intended.

The detection-model artifact consistently identified itself as `PP-OCRv6_medium_det`. The
recognition-model artifact was internally contradictory: its `inference.yml` and README
identified it as `PP-OCRv6_medium_rec`, but its own `config.json` declared
`model_type: pp_ocrv6_small_rec`. The safetensors files themselves matched their expected
SHA-256 hashes — so this was **not** file corruption or a tampering event, it was an
**artifact identity/provenance ambiguity**.

The system responded with status `ARTIFACT_IDENTITY_AMBIGUOUS` and stopped: it did not run
inference, did not silently treat the model as "Medium" (which would misrepresent the
candidate), and did not silently substitute the "Small" variant (which would have used an
unapproved model). **No production OCR change was made, and no misleading benchmark was
produced.** PP-OCRv6 does not appear in the approved model manifest and remains an
unapproved, unused candidate.

---

## 4. CURRENT ARCHITECTURE

| Layer | Current Direction | Current Status |
|---|---|---|
| Document processing | PDF/image parsing, native-text fast path for digital PDFs | Implemented |
| OCR / Vision | `Qwen/Qwen3-VL-4B-Instruct` (pinned revision) | Implemented, runtime-validated; clinical benchmark pending |
| OCR cleanup | SymSpell + regex + medical abbreviation dictionary | Implemented and tested |
| Medical NER | `d4data/biomedical-ner-all` (pinned revision, production winner) | Implemented, runtime-validated; clinical thresholds pending |
| Entity Linking | SciSpaCy + UMLS | Architecture/interface only — no approved model, UMLS license pending |
| Relations | `michiyasunaga/BioLinkBERT-base` | Backbone pinned, but artifact is an untrained base encoder — fails closed, no working relation extraction |
| Embeddings | BioClinical ModernBERT | Provider implemented; exact repo ID/revision still `PENDING_APPROVAL`, not wired to retrieval |
| Vector DB | Qdrant | Not implemented — no code exists yet |
| Simplification | `Qwen/Qwen3-0.6B` (pinned revision) | Implemented, runtime-validated; clinical faithfulness thresholds pending |
| Verification | `pritamdeka/PubMedBERT-MNLI-MedNLI` (pinned revision) | Technically validated; license pending, not production-approved |
| Translation | `ai4bharat/indictrans2-en-indic-dist-200M` (pinned revision) | Implemented, runtime-validated end-to-end |
| TTS | Kokoro TTS (target) | Not implemented — deferred for MVP |
| API | FastAPI | Implemented |
| Database | PostgreSQL (async SQLAlchemy + Alembic) | Fully coded; never run against a live PostgreSQL instance on this host |
| Cache | Redis | Fully coded; never run against a live Redis instance on this host |
| Background tasks | Celery | Fully coded; never run against a live broker on this host |
| Migrations | Alembic | One reversible migration exists; offline upgrade/downgrade SQL verified, not run live |

---

## 5. WHAT CURRENTLY WORKS

Verified directly against the repository and its own evidence, as of this report:

- FastAPI backend with independently versioned `/api/v1/*` services for OCR, NER, Entity
  Linking, Relation Extraction, Embeddings, Simplification, Verification, and Translation.
- `POST /api/v1/ocr` with real PDF/PNG/JPEG/TIFF decoding and post-processing.
- Same-origin engineering dashboards per service, plus a consolidated `/engineering-demo`
  pipeline console.
- Real local Qwen3-VL OCR inference on CPU and GPU with a pinned, checksummed checkpoint.
- Real local NER inference with `d4data/biomedical-ner-all`, correctly extracting entities
  from clinical text with genuine model confidences.
- `POST /api/v1/simplify` producing Clinical / General Public / Child-Friendly output, with a
  fail-closed grounding check that has been observed correctly rejecting a simplification that
  introduced an unsupported numeric fact.
- Real batch simplification support in the provider and service layer.
- CUDA execution with automatic device selection (`device="auto"`) across OCR, NER,
  Simplification, and Translation.
- Real IndicTrans2 translation to Hindi, Tamil, and Kannada with protected numeric/unit/dosage
  values verified intact.
- A real, no-mock, end-to-end run of Upload → OCR → NER → Simplification → Translation on this
  machine, using only pinned local models.
- Runtime observability endpoint (`GET /api/v1/runtime/metrics`) exposing real GPU/CPU memory
  and per-stage timing — fields the backend cannot measure are reported as literal
  `NOT EXPOSED`, never fabricated.
- A working artifact-identity safety gate, demonstrated by the PP-OCRv6 rejection.

**Test count:** the project's own implementation log records a progression of automated test
counts as features were added, from 70 tests (early Phase 2) up to **117 passed, 1 skipped
(CUDA-conditional), 0 failed** as of the most recent logged full-suite run (2026-08-09). This
report independently re-ran the full suite during this audit (with a writable `--basetemp`,
since the default Windows temp directory on this host intermittently denies pytest permission
to create its cache directory — a known, pre-existing environment quirk, not a code defect)
and confirmed the identical result: **117 passed, 1 skipped, 0 failed**, in 100.80s. `ruff
check` also passed with no findings, and the FastAPI app imports and initializes successfully
with 35 OpenAPI paths.

---

## 6. WHAT DID NOT WORK / WHAT CHANGED

| Item | Issue | Decision / Outcome |
|---|---|---|
| CPU-only PyTorch | A physical RTX 5050 GPU was present but unusable — PyTorch was installed as a CPU-only build | CUDA-enabled PyTorch (`2.11.0+cu128`) installed and verified in the dev environment; device defaults changed to `auto` |
| Classical OCR architecture | Earlier architecture depended on PaddleOCR/TrOCR | Architecture evolved toward Qwen3-VL vision-language OCR; classical OCR code was removed from the production path, not because it was "bad," but because requirements evolved toward handling handwriting and complex layouts in one model |
| PP-OCRv6 candidate artifact | Recognition-model metadata contradicted itself (`_rec` filename vs. `_small_rec` in config) | Candidate rejected with `ARTIFACT_IDENTITY_AMBIGUOUS`; no inference run, no production change, no silent substitution |
| "Project Vaani" | Originally considered as if it could directly serve the translation/speech runtime layer | In practice it is closer to a speech/language dataset and research ecosystem, not a drop-in translation+TTS inference stack; runtime direction is IndicTrans2 for translation and Kokoro/Bhashini/Piper (final choice pending) for TTS. Vaani may retain dataset/research value but is not current runtime architecture |
| GLiNER CPU performance | Functional but slow in CPU profiling (~12s/page, ~92s for ten pages) | Not selected for production NER; `d4data/biomedical-ner-all` was selected instead based on measured macro F1 on a small evaluation set. GPU benchmarking of GLiNER was not subsequently performed |
| QLoRA fine-tuning run | Training was started on the 5,000-example synthetic dataset but interrupted at epoch 1 of 3 | No adapter checkpoint exists; the effort is incomplete, not paused-but-successful. Production simplification uses the base Qwen3-0.6B model, not a fine-tuned adapter |
| BioLinkBERT relation extraction | The pinned checkpoint is a base `BertModel` encoder with no trained classification head, ontology, or no-relation label | Provider correctly reports `incompatible_artifact` and fails closed rather than fabricating relations; a properly fine-tuned checkpoint has not been sourced |
| Leaked Hugging Face token | `Evaluation/test.py` contains a live-looking, plaintext Hugging Face API token | Not yet rotated or removed as of this report; flagged as a hard blocker before any public GitHub push (see Section 7) |

---

## 7. PENDING REQUIREMENTS

### Models

Final validated/pinned versions and clinical-corpus thresholds are still needed for:

- Qwen3-VL OCR (checksum pending; live CER/WER benchmark pending)
- Biomedical NER (`d4data/biomedical-ner-all`) clinical-corpus thresholds
- BioClinical ModernBERT embeddings (repository ID/revision/license still `PENDING_APPROVAL`)
- SciSpaCy + UMLS entity linking (no approved artifact yet)
- A fine-tuned biomedical relation-extraction checkpoint (current pinned backbone is an
  incompatible base encoder)
- PubMedBERT MedNLI verification (technically working; license not yet cleared)
- IndicTrans2 (approved and checksummed; clinical/native-speaker review pending)
- A final TTS system selection (Kokoro is the current target direction; not yet implemented)

### Medical Knowledge / Licensing

Licensing/access must be confirmed, not assumed, for: UMLS, SNOMED CT, ICD-10 resources where
applicable, MedlinePlus, any PubMed data used, and DrugBank if used. None of these should be
treated as freely redistributable without explicit confirmation of their license terms.

### Infrastructure

PostgreSQL, Alembic (live upgrade/downgrade), Redis, Qdrant, Celery, persistent job tracking,
model caching strategy, production configuration, and containerization/deployment hardening
are all coded but unvalidated against real running services on this host.

### Evaluation

A proper medical evaluation suite is still needed, covering:

- **OCR:** CER, WER, layout/document extraction quality
- **NER:** precision, recall, F1 on a representative clinical corpus (current evaluation set
  is a 4-record synthetic sample — too small for clinical acceptance)
- **Simplification:** factual preservation, readability, terminology preservation, and
  clinician/medical review where possible
- **Verification:** entailment accuracy, contradiction detection, false-reassurance and
  omitted-information analysis
- **Translation:** medical terminology preservation and native-language human review
- **Performance:** latency, GPU/CPU memory, throughput, and batch throughput under realistic
  concurrent load (current measurements are single-request, single-host smoke tests only)

---

## 8. IMMEDIATE NEXT PRIORITIES

1. Freeze and clinically validate the OCR/VLM model against a representative, reviewed corpus.
2. Establish a reproducible OCR benchmark (CER/WER) with independently reviewed gold text.
3. Finalize the NER model using accuracy + latency comparison on a larger clinical corpus.
4. Integrate SciSpaCy/UMLS entity linking once UMLS licensing is confirmed.
5. Evaluate BioClinical ModernBERT embeddings against an approved, pinned checkpoint.
6. Build the Qdrant medical-knowledge retrieval layer (currently not started).
7. Source and validate a properly fine-tuned biomedical relation-extraction checkpoint.
8. Validate Qwen simplification quality against clinical faithfulness/readability thresholds.
9. Stand up and validate PostgreSQL/Redis/Alembic/Celery against live services.
10. Complete clinical/native-speaker review of IndicTrans2 translation output.
11. Select and integrate a final TTS provider.
12. Run a complete end-to-end medical-report evaluation across all stages together.
13. Prepare deployment and mobile/web integration interfaces.

---

## 9. CURRENT PROJECT STATUS

**Working**
- OCR, NER, Simplification, and Translation as individually validated services with real
  local model inference on CPU and GPU.
- A real, no-mock end-to-end pipeline run through all four MVP stages.
- Fail-closed safety behavior for unsupported simplification facts and for ambiguous model
  artifacts (demonstrated, not just designed).
- GPU-accelerated inference with measured, meaningful speedups.

**Partially Implemented**
- PostgreSQL/Redis/Celery (fully coded, never run live).
- Embeddings (works standalone, not connected to any retrieval store).
- Verification (technically working, not license-cleared or production-approved).

**Pending**
- Entity Linking, Relation Extraction, TTS, and Qdrant-based knowledge retrieval.
- Clinical-quality evaluation across every stage.
- Production hardening, security review, and load testing.

**Blocked / Needs Attention**
- A leaked Hugging Face token in `Evaluation/test.py` (not yet rotated).
- An incomplete QLoRA fine-tuning run with no usable checkpoint.
- A PP-OCRv6 candidate correctly rejected pending artifact-identity resolution.

The Medical Term Simplifier has progressed from architecture design into a functioning
modular prototype with document ingestion, OCR-related processing, medical NLP components,
Qwen-based simplification, API infrastructure, GPU execution, and early performance
optimization. The major remaining work concerns model validation, medical knowledge
integration, factual verification, multilingual deployment, persistent infrastructure,
systematic medical evaluation, and production hardening. It is not production-ready and
should not be represented as such.

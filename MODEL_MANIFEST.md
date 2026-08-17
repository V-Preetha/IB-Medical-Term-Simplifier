# IB Health Model Manifest

Status: engineering contract  
Schema version: 1  
Last updated: 2026-08-04

This manifest is the only repository-owned source for approved OCR and Medical NER model
repository IDs, immutable revisions, licenses, cache locations, and integrity values. Corresponding
environment variables may supply deployment-approved values. Providers must fail closed
when neither source contains an approved value.

`PENDING_APPROVAL` is intentional. It means no repository, revision, license, or checksum
has been approved. It must never be interpreted as a default, `latest`, `main`, or a
substitute checkpoint. No model may be downloaded until its repository ID and immutable
revision are approved here or supplied through the documented environment variables.

## Approved inventory

### Qwen3-VL OCR provider

| Field | Value |
| --- | --- |
| Model Name | Qwen3-VL-4B-Instruct |
| Purpose | Page-ordered OCR for supported medical document images |
| Provider | `qwen3-vl` |
| Repository ID | `Qwen/Qwen3-VL-4B-Instruct` |
| Pinned Revision | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| License | `Apache-2.0` |
| Local Cache Path | `New_current/.model-cache/qwen3-vl` |
| Expected SHA256 | `PENDING_APPROVAL` |
| Device | `auto` (selects CUDA when available, otherwise CPU; explicit `cpu`/`cuda` overrides fail closed rather than silently falling back) |
| Configuration Variables | `OCR_CONFIG__MODEL_NAME`, `OCR_CONFIG__MODEL_REVISION`, `OCR_CONFIG__HF_CACHE_DIR`, `OCR_CONFIG__DEVICE` |

This is the sole approved Phase 2 model. Another Qwen family, size, quantization, or
moving revision must not be substituted. Document type, when required, is inferred in
the same Qwen3-VL generation using the versioned structured prompt contract.

## Machine-readable contract

The application and validation runner parse only the JSON object between these markers.
Human-readable tables above must remain consistent with it.

<!-- MODEL_MANIFEST_DATA_BEGIN -->
```json
{
  "schema_version": "1",
  "models": {
    "qwen3-vl": {
      "model_name": "Qwen3-VL-4B-Instruct",
      "purpose": "Page-ordered OCR for supported medical document images",
      "provider": "qwen3-vl",
      "repository_id": "Qwen/Qwen3-VL-4B-Instruct",
      "pinned_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
      "license": "Apache-2.0",
      "local_cache_path": "New_current/.model-cache/qwen3-vl",
      "expected_sha256": "PENDING_APPROVAL",
      "device": "auto",
      "configuration_variables": [
        "OCR_CONFIG__MODEL_NAME",
        "OCR_CONFIG__MODEL_REVISION",
        "OCR_CONFIG__HF_CACHE_DIR",
        "OCR_CONFIG__DEVICE"
      ]
    }
  }
}
```
<!-- MODEL_MANIFEST_DATA_END -->

## Phase 5 Medical NER inventory

`d4data/biomedical-ner-all` is the approved production winner. OpenMed GLiNER and
ModernBERT are archived benchmark references and are excluded from the production registry,
dependency injection, lifecycle, health, model inventory API, and startup. All providers
remain local-only and never resolve a moving revision or trigger a download.

| Candidate | Status | Framework | Repository | Revision | License | Local cache |
| --- | --- | --- | --- | --- | --- | --- |
| OpenMed GLiNER (Zero-Shot) | Archived runner-up | GLiNER | `OpenMed/OpenMed-ZeroShot-NER-Pathology-Medium-209M` | `e63d8b131d599970674d05617bdbd1a3eef495ee` | `Apache-2.0` | `New_current/.model-cache/ner/openmed-gliner` |
| biomedical-ner-all | **Winner Approved - Production** | Transformers | `d4data/biomedical-ner-all` | `015a4050c9ac99722e61c547aa9b4282bcbedc7f` | `Apache-2.0` | `New_current/.model-cache/ner/biomedical-ner-all` |
| Kushtrim/ModernBERT-base-biomedical-ner | Archived runner-up | Transformers | `Kushtrim/ModernBERT-base-biomedical-ner` | `a4ebc00ce8c52ac03ccaaae96600431b9e4c3e39` | `Apache-2.0` | `New_current/.model-cache/ner/modernbert-biomedical` |

<!-- NER_MODEL_MANIFEST_DATA_BEGIN -->
```json
{
  "schema_version": "1",
  "production_provider": "biomedical-ner-all",
  "candidates": {
    "openmed-gliner": {
      "model_name": "OpenMed GLiNER (Zero-Shot)",
      "purpose": "Zero-shot medical named entity recognition evaluation",
      "provider": "openmed-gliner",
      "framework": "gliner",
      "repository_id": "OpenMed/OpenMed-ZeroShot-NER-Pathology-Medium-209M",
      "pinned_revision": "e63d8b131d599970674d05617bdbd1a3eef495ee",
      "license": "Apache-2.0",
      "local_cache_path": "New_current/.model-cache/ner/openmed-gliner",
      "expected_sha256": "PENDING_APPROVAL",
      "configuration_variables": [
        "NER_OPENMED_GLINER__REPOSITORY_ID",
        "NER_OPENMED_GLINER__REVISION",
        "NER_OPENMED_GLINER__CACHE_DIR",
        "NER_OPENMED_GLINER__LABEL_MAPPING_JSON"
      ]
    },
    "biomedical-ner-all": {
      "model_name": "biomedical-ner-all",
      "purpose": "Medical named entity token-classification evaluation",
      "provider": "biomedical-ner-all",
      "framework": "transformers-token-classification",
      "repository_id": "d4data/biomedical-ner-all",
      "pinned_revision": "015a4050c9ac99722e61c547aa9b4282bcbedc7f",
      "license": "Apache-2.0",
      "local_cache_path": "New_current/.model-cache/ner/biomedical-ner-all",
      "expected_sha256": "PENDING_APPROVAL",
      "configuration_variables": [
        "NER_CONFIG__PROVIDER",
        "NER_CONFIG__MODEL_NAME",
        "NER_CONFIG__MODEL_REVISION",
        "NER_CONFIG__CACHE_DIR",
        "NER_CONFIG__DEVICE",
        "NER_CONFIG__CONFIDENCE_THRESHOLD",
        "NER_CONFIG__MAX_TOKENS",
        "NER_CONFIG__STRIDE_TOKENS",
        "NER_CONFIG__LABEL_MAPPING_JSON"
      ]
    },
    "modernbert-biomedical-ner": {
      "model_name": "Kushtrim/ModernBERT-base-biomedical-ner",
      "purpose": "ModernBERT medical named entity token-classification evaluation",
      "provider": "modernbert-biomedical-ner",
      "framework": "transformers-token-classification",
      "repository_id": "Kushtrim/ModernBERT-base-biomedical-ner",
      "pinned_revision": "a4ebc00ce8c52ac03ccaaae96600431b9e4c3e39",
      "license": "Apache-2.0",
      "local_cache_path": "New_current/.model-cache/ner/modernbert-biomedical",
      "expected_sha256": "PENDING_APPROVAL",
      "configuration_variables": [
        "NER_MODERNBERT_BIOMEDICAL_NER__REPOSITORY_ID",
        "NER_MODERNBERT_BIOMEDICAL_NER__REVISION",
        "NER_MODERNBERT_BIOMEDICAL_NER__CACHE_DIR",
        "NER_MODERNBERT_BIOMEDICAL_NER__LABEL_MAPPING_JSON"
      ]
    }
  }
}
```
<!-- NER_MODEL_MANIFEST_DATA_END -->

## Entity Linking production inventory

SciSpaCy with the UMLS Entity Linker is the approved Phase 6 architecture. Exact package,
language-model, terminology release, local artifact, and license identities have not yet
been approved or provisioned in this workspace. They remain explicit rather than being
guessed. Runtime initialization therefore fails closed until every value is supplied in
this manifest or its corresponding environment variable and the licensed artifacts are
available locally.

<!-- ENTITY_LINKING_MANIFEST_DATA_BEGIN -->
```json
{
  "schema_version": "1",
  "production": {
    "provider": "scispacy-umls",
    "provider_version": "PENDING_APPROVAL",
    "language_model": "PENDING_APPROVAL",
    "language_model_version": "PENDING_APPROVAL",
    "language_model_path": "PENDING_APPROVAL",
    "terminology": "UMLS",
    "terminology_version": "PENDING_APPROVAL",
    "knowledge_base_path": "PENDING_APPROVAL",
    "license": "PENDING_APPROVAL",
    "configuration_variables": [
      "ENTITY_LINKING_CONFIG__PROVIDER",
      "ENTITY_LINKING_CONFIG__SCISPACY_VERSION",
      "ENTITY_LINKING_CONFIG__LANGUAGE_MODEL",
      "ENTITY_LINKING_CONFIG__LANGUAGE_MODEL_VERSION",
      "ENTITY_LINKING_CONFIG__LANGUAGE_MODEL_PATH",
      "ENTITY_LINKING_CONFIG__UMLS_RELEASE",
      "ENTITY_LINKING_CONFIG__UMLS_KB_PATH",
      "ENTITY_LINKING_CONFIG__UMLS_LICENSE_ACCEPTED",
      "ENTITY_LINKING_CONFIG__CONFIDENCE_THRESHOLD",
      "ENTITY_LINKING_CONFIG__MAX_CANDIDATES",
      "ENTITY_LINKING_CONFIG__AMBIGUITY_DELTA"
    ]
  }
}
```
<!-- ENTITY_LINKING_MANIFEST_DATA_END -->

## Phase 7 Biomedical Relation Extraction inventory

`michiyasunaga/BioLinkBERT-base` is the approved Apache-2.0 BioLinkBERT backbone. The
local Hugging Face artifact metadata records immutable revision
`b71f5d70f063d1c8f1124070ce86f1ee463ca1fe`. Its current `config.json` declares a base
`BertModel`, not a trained sequence-classification relation head. Production inference
must therefore remain fail-closed until an approved relation-classification artifact at
this identity supplies named `id2label`, `no_relation_labels`, and the matching
`entity-marker-v1` preprocessing declaration. Transformers must never initialize a random
classification head for production use.

| Field | Value |
| --- | --- |
| Model Name | BioLinkBERT-base |
| Purpose | Biomedical relation extraction backbone |
| Provider | `biolinkbert` |
| Repository ID | `michiyasunaga/BioLinkBERT-base` |
| Pinned Revision | `b71f5d70f063d1c8f1124070ce86f1ee463ca1fe` |
| License | `Apache-2.0` |
| Local Cache Path | `New_current/.model-cache/relation-extraction/biolinkbert` |
| Expected SHA256 | `PENDING_APPROVAL` |
| Preprocessing | `entity-marker-v1` |
| Confidence | Uncalibrated relation-class softmax probability |

<!-- RELATION_EXTRACTION_MANIFEST_DATA_BEGIN -->
```json
{
  "schema_version": "1",
  "production": {
    "model_name": "BioLinkBERT-base",
    "purpose": "Biomedical relation extraction",
    "provider": "biolinkbert",
    "framework": "transformers-sequence-classification",
    "repository_id": "michiyasunaga/BioLinkBERT-base",
    "pinned_revision": "b71f5d70f063d1c8f1124070ce86f1ee463ca1fe",
    "license": "Apache-2.0",
    "local_cache_path": "New_current/.model-cache/relation-extraction/biolinkbert",
    "expected_sha256": "PENDING_APPROVAL",
    "preprocessing_version": "entity-marker-v1",
    "calibration_version": "uncalibrated-biolinkbert-re-v1",
    "configuration_variables": [
      "RELATION_CONFIG__PROVIDER",
      "RELATION_CONFIG__MODEL_NAME",
      "RELATION_CONFIG__MODEL_REVISION",
      "RELATION_CONFIG__CACHE_DIR",
      "RELATION_CONFIG__DEVICE",
      "RELATION_CONFIG__ALLOW_CPU_FALLBACK",
      "RELATION_CONFIG__CONFIDENCE_THRESHOLD",
      "RELATION_CONFIG__BATCH_SIZE",
      "RELATION_CONFIG__MAX_LENGTH",
      "RELATION_CONFIG__MAX_ENTITY_PAIRS",
      "RELATION_CONFIG__PREPROCESSING_VERSION",
      "RELATION_CONFIG__CALIBRATION_VERSION",
      "RELATION_CONFIG__NO_RELATION_LABELS_JSON"
    ]
  }
}
```
<!-- RELATION_EXTRACTION_MANIFEST_DATA_END -->

## Medical Embeddings inventory

BioClinical ModernBERT is the approved embedding architecture. The exact repository ID,
immutable revision, license, and local cache have not been approved or provisioned in the
production source boundary. Experimental references elsewhere in the repository are not
promoted into production. Runtime therefore fails closed until deployment supplies the
approved values through this manifest or the corresponding environment variables.

<!-- EMBEDDING_MANIFEST_DATA_BEGIN -->
```json
{
  "schema_version": "1",
  "production": {
    "model_name": "BioClinical ModernBERT",
    "purpose": "Contextual medical text embedding generation",
    "provider": "bioclinical-modernbert",
    "framework": "transformers-feature-extraction",
    "repository_id": "PENDING_APPROVAL",
    "pinned_revision": "PENDING_APPROVAL",
    "license": "PENDING_APPROVAL",
    "local_cache_path": "PENDING_APPROVAL",
    "expected_sha256": "PENDING_APPROVAL",
    "pooling_method": "attention-mask-mean-v1",
    "configuration_variables": [
      "EMBEDDING_CONFIG__PROVIDER",
      "EMBEDDING_CONFIG__MODEL_NAME",
      "EMBEDDING_CONFIG__MODEL_REVISION",
      "EMBEDDING_CONFIG__LICENSE",
      "EMBEDDING_CONFIG__CACHE_DIR",
      "EMBEDDING_CONFIG__DEVICE",
      "EMBEDDING_CONFIG__ALLOW_CPU_FALLBACK",
      "EMBEDDING_CONFIG__BATCH_SIZE",
      "EMBEDDING_CONFIG__MAX_LENGTH",
      "EMBEDDING_CONFIG__NORMALIZE",
      "EMBEDDING_CONFIG__POOLING_METHOD"
    ]
  }
}
```
<!-- EMBEDDING_MANIFEST_DATA_END -->

## MVP Qwen3 Simplification inventory

| Field | Value |
| --- | --- |
| Provider | `qwen3` |
| Repository ID | `Qwen/Qwen3-0.6B` |
| Pinned Revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| License | `Apache-2.0` |
| Prompt Version | `qwen-medical-simplification-v2` |
| Runtime Policy | Local files only; deterministic generation; no production fallback |

The immutable revision is evidenced by the locally cached snapshot. The provider requests
that exact revision and fails closed when it is absent.

<!-- SIMPLIFICATION_MANIFEST_DATA_START -->
```json
{
  "manifest_version": "simplification-model-manifest-v1",
  "production_model": {
    "purpose": "Three-level medical report simplification",
    "provider": "qwen3",
    "repository_id": "Qwen/Qwen3-0.6B",
    "pinned_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    "license": "Apache-2.0",
    "local_cache_path": "HF_HUB_CACHE/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
    "prompt_version": "qwen-medical-simplification-v2",
    "prompt_path": "New_current/app/simplification/prompts/medical_report_v2.json",
    "device": "auto with explicit CPU fallback",
    "configuration_variables": [
      "SIMPLIFICATION_CONFIG__MODEL_ID",
      "SIMPLIFICATION_CONFIG__MODEL_REVISION",
      "SIMPLIFICATION_CONFIG__MODEL_PATH",
      "SIMPLIFICATION_CONFIG__DEVICE",
      "SIMPLIFICATION_CONFIG__MAX_INPUT_CHARACTERS",
      "SIMPLIFICATION_CONFIG__MAX_NEW_TOKENS"
    ],
    "runtime_policy": "local_files_only"
  }
}
```
<!-- SIMPLIFICATION_MANIFEST_DATA_END -->

## MVP IndicTrans2 Translation inventory

| Field | Value |
| --- | --- |
| Provider | `indictrans2` |
| Repository ID | `ai4bharat/indictrans2-en-indic-dist-200M` |
| Pinned Revision | `173b94239f7c38886b2747b8d4a5db771a7e1232` |
| License | `MIT` |
| Local Cache Path | `New_current/.model-cache/translation/indictrans2-en-indic-dist-200M` |
| Primary Weight Artifact | `model.safetensors` |
| SHA256 | `0039e4304e9889acc5c8350a193311d07aa5399b6d9fb0445e8fde19e3533bb5` |
| Preservation Policy | `numeric-unit-bracketed-placeholders-v2` |
| Runtime Policy | Local files only; no heuristic or remote fallback |

The immutable model identity and primary weight checksum are approved. The provider verifies
the SHA-256 of this exact local `model.safetensors` artifact before initialization and loads
only from this directory. It does not use `pytorch_model.bin` unless a separately approved
compatibility change is recorded.

<!-- TRANSLATION_MODEL_MANIFEST_DATA_BEGIN -->
```json
{
  "schema_version": "1",
  "production_model": {
    "model_name": "IndicTrans2 English-to-Indic Distilled 200M",
    "provider": "indictrans2",
    "repository_id": "ai4bharat/indictrans2-en-indic-dist-200M",
    "pinned_revision": "173b94239f7c38886b2747b8d4a5db771a7e1232",
    "license": "MIT",
    "local_cache_path": "New_current/.model-cache/translation/indictrans2-en-indic-dist-200M",
    "expected_sha256": "0039e4304e9889acc5c8350a193311d07aa5399b6d9fb0445e8fde19e3533bb5",
    "configuration_variables": [
      "TRANSLATION_CONFIG__MODEL_ID",
      "TRANSLATION_CONFIG__MODEL_REVISION",
      "TRANSLATION_CONFIG__MODEL_PATH",
      "TRANSLATION_CONFIG__DEVICE",
      "TRANSLATION_CONFIG__MODULE_CACHE_PATH",
      "TRANSLATION_CONFIG__MAX_NEW_TOKENS"
    ]
  }
}
```
<!-- TRANSLATION_MODEL_MANIFEST_DATA_END -->

## Medical Verification technical candidate

| Field | Value |
| --- | --- |
| Repository | pritamdeka/PubMedBERT-MNLI-MedNLI |
| Pinned Revision | f1b6ce2e0d49f295b4cbcdc56c01b5fab6d068ab |
| Local Cache | New_current/.model-cache/verification/pubmedbert-mednli |
| Architecture | BertForSequenceClassification |
| Label Mapping | 0=contradiction, 1=entailment, 2=neutral |
| Maximum Sequence Length | 512 |
| License | PENDING_VERIFICATION |
| Status | TECHNICALLY VERIFIED; PRODUCTION APPROVAL PENDING |

This checkpoint may be used only for technical validation. License verification is required
before production approval.

<!-- VERIFICATION_MODEL_MANIFEST_DATA_BEGIN -->
json
{
  "repository_id": "pritamdeka/PubMedBERT-MNLI-MedNLI",
  "pinned_revision": "f1b6ce2e0d49f295b4cbcdc56c01b5fab6d068ab",
  "license": "PENDING_VERIFICATION",
  "local_cache_path": "New_current/.model-cache/verification/pubmedbert-mednli"
}
<!-- VERIFICATION_MODEL_MANIFEST_DATA_END -->

## Approval procedure

Approval must replace each `PENDING_APPROVAL` repository ID and revision with the exact
upstream identifier and immutable commit SHA, record the checkpoint license, and record an
expected SHA256 when an authoritative artifact hash is available. Validation must then
confirm the configured values, local snapshot revision, and available checksum before live
inference is reported as verified.

When `expected_sha256` is supplied for a snapshot directory, it is the lowercase SHA-256
of each file's POSIX-style relative path followed by its bytes, with files processed in
lexicographic relative-path order. This is the same deterministic tree-digest algorithm
used by the validation runner.

# PP-OCRv6 Medium Artifact Validation

Status: `ARTIFACT_IDENTITY_AMBIGUOUS`

Date: 2026-08-12

## Scope

This is an isolated benchmark-artifact validation. No production OCR configuration, API, or
provider implementation was changed. No network request, automatic PaddleOCR download, or
model substitution was attempted.

## Integrity verification

| Component | Expected SHA-256 | Observed SHA-256 | Status |
| --- | --- | --- | --- |
| Detection `model.safetensors` | `BD393266C02E1A680B1B34C301D5D0D81E6290440B7F8AB0F5D5032276B17EB1` | `BD393266C02E1A680B1B34C301D5D0D81E6290440B7F8AB0F5D5032276B17EB1` | PASS |
| Recognition `model.safetensors` | `5F43C16F2A684B1D2284662178BDB604FEBD3D6BFDB5CA73828D08D0F7C0C3E9` | `5F43C16F2A684B1D2284662178BDB604FEBD3D6BFDB5CA73828D08D0F7C0C3E9` | PASS |

Both artifacts contain the required `config.json`, `inference.yml`,
`preprocessor_config.json`, and `README.md` files.

## Identity validation

| Component | Evidence | Result |
| --- | --- | --- |
| Detection | `config.json` has `model_type: pp_ocrv6_medium_det`; `inference.yml` has `model_name: PP-OCRv6_medium_det`. | Consistent Medium detection identity |
| Recognition | `inference.yml` has `model_name: PP-OCRv6_medium_rec`; README heading is `PP-OCRv6_medium_rec`. | Medium claims present |
| Recognition | Actual downloaded `config.json` has `model_type: pp_ocrv6_small_rec`. | Contradicts Medium claims |

The recognition model’s internal configuration identifies it as **small**, despite the
repository name, README, and inference metadata identifying it as **medium**. A matching
checksum establishes that the provisioned artifact was not corrupted or changed after
approval; it does not resolve this conflicting model identity.

## Decision

`ARTIFACT_IDENTITY_AMBIGUOUS`

No DET/REC initialization, PDF Deid smoke test, Qwen comparison, hybrid evaluation, or
benchmark recommendation was performed. Loading it as Medium would misrepresent the tested
candidate; loading it as Small would substitute a non-approved candidate. The required next
action is explicit artifact-owner resolution of the recognition checkpoint identity, followed
by a corrected immutable approval or an approval that explicitly authorizes this artifact as
the intended candidate.

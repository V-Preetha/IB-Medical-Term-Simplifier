# Phase 9 Medical Report Simplification Integration Report

Date: 2026-08-06  
Disposition: **IN PROGRESS - clinical validation remains open**

## Production boundary

The service exposes the approved local `Qwen/Qwen3-0.6B` checkpoint through a provider
interface, application service, dependency injection, typed errors, health/model APIs,
structured privacy-safe logs, OpenAPI, and a same-origin dashboard. The immutable revision
is `c1899de289a04d12100db370d81485cdf75e47ca`; model files are loaded locally only, with no
cloud API or alternate-model fallback.

One deterministic inference returns Clinical, General Public, and Child-Friendly versions.
Every version includes source, simplified report, term explanations, important findings,
suggested clinician questions, fidelity confidence, latency, model revision, pipeline
version, prompt version, warnings, and review state.

## Safety controls

Prompt `qwen-medical-simplification-v2` is externalized under
`app/simplification/prompts/`. It marks source text as untrusted data and prohibits new
diagnoses, medications, values, advice, prognosis, and unsupported facts. The service
independently rejects numeric values/units absent from source and explanations for terms
not in the source evidence. Confidence is an uncalibrated source-fact/entity preservation
ratio, not a probability of clinical correctness. All output requires review.

## Verification evidence

| Gate | Result | Evidence |
|---|---|---|
| Pinned local initialization | PASS | CPU healthy; 3758.256 ms; exact revision |
| Three-level real inference | PASS | Negated synthetic report; 91078.880 ms; 451 output tokens |
| Preservation score | PASS for validated sample | 1.0 for all three levels |
| Unsupported numeric rejection | PASS | Typed safety rejection in tests and real HbA1c run |
| Numeric/laboratory generation | NOT VERIFIED | Real HbA1c output added a number and was rejected |
| CUDA inference | NOT VERIFIED | No CUDA device in installed runtime |
| Ruff | PASS | Repository-wide check |
| Complete tests | PASS | 86 passed, 1 CUDA-conditional skip |
| Compilation/imports | PASS | Compilation and application test imports |
| Swagger/dashboard | PASS | Exact operations, schemas, page, assets, JavaScript syntax |
| Clinical/readability benchmark | NOT VERIFIED | Approved corpus/thresholds not executed |

## Recommendation

The boundary is suitable for controlled engineering evaluation, but Phase 9 must not be
marked complete or used for unsupervised patient output. CPU latency is high and the
numeric-report rejection frequency is unknown. Run the approved de-identified clinical
benchmark and resolve that gate without weakening source-grounding controls.

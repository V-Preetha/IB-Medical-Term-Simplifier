# OCR Output Contract Optimization Experiment

Status: NOT VERIFIED — no production change

## Controlled configuration

The run used only the pinned local `Qwen/Qwen3-VL-4B-Instruct` revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`, BF16, and `device=auto`. Both contracts use
the same 512px / 96-DPI benchmark profile, deterministic decoding, `max_new_tokens=2048`,
and a 300-second generation policy. This diagnostic profile is not the frozen production
1600px / 144-DPI profile.

The candidate requests transcription only and constructs the response envelope in Python.
The public API, production prompt, model identity/revision, post-processing implementation,
and production defaults are unchanged. One model/tokenizer is reused by both contracts.

## Real-run outcome

| Requirement | Evidence | Status |
| --- | --- | --- |
| Pinned BF16 model initialization | Local checkpoint shards loaded before inference. | PASS |
| 2 Medium + 2 Hard comparison | First selected multi-page Medium structured run had no final result after more than six minutes; the process was terminated. | NOT VERIFIED |
| Baseline / candidate latency, token, PHI metrics | No completed sequence or candidate run exists. | NOT VERIFIED |
| Termination reliability | Benchmark-only 300-second stopping is implemented; no completed four-document run has finished with it. | NOT VERIFIED |
| Same production post-processing resources | Local production mount is absent; symmetric fixture resources do not validate production. | NOT VERIFIED |

The earlier 45-second diagnostic remains observation-only: structured output generated 66
tokens at 1.424/1.433 tok/s, versus transcription-only 134/143 at 2.962/3.141 tok/s on one
Medium and one Hard PDF. It does not prove fidelity or qualify the candidate for promotion.

## Quality, confidence, and document type

The corpus provides PHI-token lists rather than full reviewed transcription, so CER/WER are
NOT VERIFIED. The candidate reports `mean_transcription_token_probability`; baseline reports
`mean_generated_token_probability` including JSON-envelope tokens. They are not comparable
calibrated scores.

Option A is the lowest-cost safe scanned/image strategy: `document_type=unknown`, requiring
review. Option C remains the native digital-PDF path only. Option B was not run because it
adds serial model generation.

## Recommendation and migration plan

REJECT FOR PROMOTION. Complete the controlled 2-Medium/2-Hard run with PHI and numeric
comparison before any runtime change. If separately approved after validation: introduce a
versioned environment-backed output-contract setting; route normal provider processing to
transcription-only generation; build the existing envelope through `OCRResultBuilder`; record
unknown scanned/image type as review-required; retain digital-PDF behavior; then add API and
corpus regression tests and obtain clinical approval.

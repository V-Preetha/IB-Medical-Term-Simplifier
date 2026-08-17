# Targeted Qwen Performance Validation

Date: 2026-08-09

## Scope

This validation used the approved local CUDA artifacts only:

- OCR: `Qwen/Qwen3-VL-4B-Instruct` at `ebb281ec70b05090aa6165b016eac8ec08e71b17`.
- Simplification: `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`.

No model identity, safety policy, public API, NER, Translation, Verification, database,
Redis, or Celery code was changed.

## Existing runtime controls verified

The active simplification provider already uses `model.eval()`,
`torch.inference_mode()`, deterministic greedy decoding (`do_sample=False`,
`num_beams=1`), `use_cache=True`, CUDA autocast, a process-wide loaded-model cache,
one prompt tokenization, and one structured generation containing all three readability
levels. The active OCR provider already uses `model.eval()`, `torch.inference_mode()`,
deterministic greedy decoding, per-request GPU peak-memory measurement, and one generation
that returns document type and transcription.

## Real CUDA measurements

| Stage | Existing warm baseline | Device | Evidence |
| --- | ---: | --- | --- |
| Qwen3-VL OCR plus post-processing | 7,123.953 ms | CUDA | `PERFORMANCE_PROFILE.md` |
| Qwen3 simplification | 14,264.999 ms | CUDA | `PERFORMANCE_PROFILE.md` |

Targeted de-identified fixture runs used
`benchmarks/ocr/reports/synthetic_inputs/synthetic.png` (900 x 220) and excluded
model loading from inference timing.

| Optimization | Stage | Baseline latency | Candidate latency | Speedup | Quality/safety result | Production recommendation |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Bound longest image edge to 512 px | OCR | 13,703.042 ms warm at 900 px | 7,091.040 ms warm at 512 px | 1.93x | Fixture retained `BP 120/80` and `haemoglobin 6.5 %`; only spacing differed. One synthetic image is not a representative scan/table benchmark. | Do not promote yet |
| Set OCR generation ceiling to 64 | OCR | 40 generated tokens at 128-token ceiling | 39 generated tokens at 64 | Not isolated from resolution candidate | Complete fixture JSON/transcription; little headroom. | Do not promote independently |
| Set simplification ceiling to 800 | Simplification | 760 tokens at 1,800 limit | 760 tokens at 800 limit | No latency reduction | Complete JSON; no-numeric case passed source grounding, while numeric case was correctly rejected for unsupported generated fact. | Do not promote as latency optimization |
| Set simplification ceiling to 700 | Simplification | 760-token valid numeric result | Invalid/truncated JSON | N/A | Rejected fail-closed. | Rejected |

## Simplification profile

| Measurement | Observed value |
| --- | ---: |
| Model loading time (isolated CUDA run) | 3,011.929 ms |
| Prompt tokens, numeric test | 513 |
| Generated tokens, numeric test | 760 |
| Generation latency, numeric test at 1,800 cap | 35,252.204 ms |
| Prompt tokens, no-numeric grounding-pass test | 470 |
| Generated tokens, no-numeric grounding-pass test at 800 cap | 319 |
| Generation latency, no-numeric grounding-pass test | 19,667.710 ms |

Autoregressive output length dominates. All three levels are already generated in one
call; splitting them would be slower. The active provider does not move generated tensors
to CPU until decoding after generation. The installed CUDA runtime selected BF16
automatically. BF16-versus-FP16 and compile promotion are not reported because no
output-fidelity comparison has been completed.

## OCR profile and decision

The synthetic OCR fixture generated 40 tokens at 900 px and 39 at 512 px, so the candidate
improvement is attributable primarily to fewer visual tokens/prefill work, not the output
limit. Digital PDFs continue to use the native-text fast path when a usable text layer is
available.

No new production optimization is promoted in this pass. A versioned representative,
de-identified OCR quality corpus—including tables and small text—is required before
deploying the 512-pixel candidate. The simplification cap remains 1,800 until clinical
benchmarking proves a lower ceiling yields complete source-grounded output for all required
report classes.

The fastest evidence-supported real warm E2E result remains 22,185.073 ms. With the
current approved stack on this Windows CUDA host, the Qwen stages are the fundamental
user-visible bottlenecks. If product latency requires a material reduction, a future
architecture decision must benchmark a smaller or distilled OCR or generation model; none
is recommended or performed here.

# PDF Deid Synthetic OCR Benchmark

Status: DEPLOYABLE PRECISION INITIALIZATION VERIFIED; FULL-QUALITY SMOKE BLOCKED

Dataset: pdf_deid_synthetic_medical_v1. It is a fully synthetic, medical-style PDF corpus:
30 Easy PDFs, 10 Medium PDFs, and 10 Hard PDFs. The mapping files provide filename-keyed
PHI token lists only. They do not provide complete reviewed document transcriptions, so
CER, WER, exact document-text agreement, medication recovery, and clinical-value recovery
are NOT VERIFIED.

The Easy sample uses the provider native-PDF text-layer fast path. Baseline and 512px
candidate are therefore identical for the OCR-resolution experiment; its observed candidate
latency was 23.155 ms, document type digital_pdf, three pages, and PHI unique-token
recovery 0.947368. This is synthetic PHI token recovery, not OCR accuracy.

The prior 128-token smoke configuration was incorrect and has been superseded. The resolved
Compose baseline uses max_new_tokens 2048, max_image_size 1600, PDF render DPI 144, and
dtype float32. The first Medium PDF did not reach inference: initialization failed while
moving the pinned Qwen3-VL-4B FP32 model to CUDA with CUDA out of memory on the 8GB GPU.
This is a production-baseline failure, not a 512px candidate failure.

Because the production baseline cannot initialize on this host, the corrected Easy/Medium/
Hard smoke comparison, 512px candidate, full 50-PDF run, aggregate latency, failure
shortlist, and promotion decision are NOT VERIFIED.

Hardware precision qualification:

- FP32 is NOT DEPLOYABLE ON THE CURRENT 8GB GPU: initialization fails with CUDA OOM.
- BF16 initialized on CUDA in 24,058.935 ms, with 8,519.605 MiB allocated and 8,556.000
  MiB reserved GPU memory; RSS was 1,542.270 MiB.
- FP16 initialized on CUDA in 28,244.485 ms, with 8,519.605 MiB allocated and 8,556.000
  MiB reserved GPU memory; RSS was 7,398.590 MiB.

BF16 is the preferred precision profile for this hardware, but it is not yet promoted for
production: the first full-quality BF16 smoke used the unchanged 1600px, 144 DPI,
2048-token baseline and did not complete Medium/Hard image OCR within the 360-second
harness limit. The provider therefore produced no completed baseline record for a controlled
512px comparison. Post-processing resources are also not mounted in this local benchmark
environment and remain NOT VERIFIED.

No production OCR setting was changed. The 512px candidate and full 50-PDF benchmark remain
blocked until the deployable BF16 full-quality smoke completes with the production timeout
behavior and mounted production post-processing resources.

## BF16 staged image-cost sweep

The reference 1600px at 144 DPI profile remains NOT DEPLOYABLE ON CURRENT HARDWARE because
it did not complete Medium/Hard OCR within the prior 360-second harness limit. The following
BF16 Medium-PDF sweep retained the exact pinned model, prompt, decoding, 2048-token ceiling,
and safety behavior:

| Profile | Result | Interpretation |
| --- | --- | --- |
| 1024px at 144 DPI | Initialized; no completed OCR result within 150 seconds | TOO SLOW |
| 512px at 96 DPI | Initialized; no completed OCR result within 150 seconds | TOO SLOW |

Neither run produced a structured response, generated-token count, PHI recovery result, or
clinical-text output before the practical harness limit. Therefore no resolution profile is
currently DEPLOYABLE, no 512px quality comparison exists, and the full 50-PDF run must not
start. The matching timeouts at the largest and smallest staged image profiles indicate that
the 2048-token generation behavior is the dominant unresolved cost; this requires a separate,
approved generation-limit or generation-mode investigation before image-bound selection can
be qualified.

Precision decision: MORE_VALIDATION_REQUIRED. 512px decision: MORE_VALIDATION_REQUIRED.
Model replacement should be considered only after the approved model's generation behavior
has been profiled with actual token counts; no replacement is selected here.

## Generation behavior diagnostic

Benchmark-only diagnostics used the actual loaded BF16 provider/model at 512px and 96 DPI,
the unchanged 2048-token ceiling, deterministic decoding, and a 45-second stopping
criterion. This prevents indefinite generation without lowering the configured ceiling.

| Document | Contract | Prompt tokens | Generated tokens at stop | Tokens/sec | EOS | JSON complete |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Medium | production structured document type plus JSON text | 258 | 66 | 1.424 | no | no |
| Hard | production structured document type plus JSON text | 258 | 66 | 1.433 | no | no |
| Medium | transcription only | 201 | 134 | 2.962 | no | not applicable |
| Hard | transcription only | 201 | 143 | 3.141 | no | not applicable |

The structured runs began valid JSON and emitted ordinary transcription inside the text
field; no repeated JSON fragments, repeated transcription, explanatory text, or malformed
recovery loop was observed. The transcription-only runs emitted ordinary visible text. Both
contracts were still transcribing at the 45-second stop, so the 2048-token ceiling was not
reached and EOS did not occur. The model generation configuration declares EOS token IDs
151645 and 151643, consistent with the tokenizer/chat runtime; the absence of EOS is
explained by incomplete transcription, not a demonstrated termination-token defect.

Visual processor tensors had shape 704 by 1536 for the structured runs; rendered page
dimensions were 362 by 512. Processor preparation took 127.746 ms (Medium) and 30.442 ms
(Hard), so processor work is not the dominant latency. Generation is the bottleneck.

The next recommended action is B: simplify the OCR output contract in a separate approved
experiment. Have Qwen3-VL produce transcription only, then construct the response envelope
deterministically in application code. This does not remove document type from production,
does not change it now, and requires a quality benchmark before any promotion.

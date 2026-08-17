# Performance optimization report

## Scope and invariants

This change affects backend execution only. OCR routing and engines, manual review,
public request/response schemas, frontend files, GLiNER/Qwen model choices, clinical
labels, lab parsing rules, and generated section structure are unchanged.

Both models are constructed from the FastAPI lifespan in `app/main.py`. Their
`load_once` caches are protected by process-wide locks and keyed by model/runtime
configuration, so each worker process loads each configured model once and reuses it
for every request.

## Benchmark method

- Machine: CPU-only PyTorch, 16 inference threads, 23.7 GB RAM.
- Models: `Ihor/gliner-biomed-large-v1.0` and `Qwen/Qwen3-0.6B`.
- Input: 1, 4, and 10 pages derived from the checked-in four-page cardio report.
- Model startup is excluded because production loads models during FastAPI startup.
- "Before" uses serial GLiNER chunks and the full normalized report as Qwen evidence.
- "After" uses GLiNER batch inference and compact structured Qwen evidence.
- Peak RAM is process RSS with both models loaded. Batched inference deliberately
  trades additional temporary activation memory for lower latency.

### Measured GLiNER and context construction

| Pages | Mode | Measured stage total | GLiNER | Peak RAM | Avg chunk | Chunks | Qwen evidence |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Before | 11.93 s | 11.91 s | 2,614 MB | 1,461 chars | 2 | 2,921 chars |
| 1 | After | 12.84 s | 12.82 s | 2,882 MB | 1,461 chars | 2 | 2,134 chars |
| 4 | Before | 44.75 s | 44.71 s | 2,723 MB | 1,866 chars | 6 | 11,194 chars |
| 4 | After | 36.46 s | 36.40 s | 3,393 MB | 1,866 chars | 6 | 6,608 chars |
| 10 | Before | 115.07 s | 114.99 s | 2,665 MB | 1,893 chars | 15 | 28,396 chars |
| 10 | After | 92.19 s | 92.05 s | 3,492 MB | 1,893 chars | 15 | 6,462 chars |

On this CPU, batching has a small two-chunk overhead for one page, then reduces GLiNER
latency by 18.6% at four pages and 20.0% at ten pages. The Qwen evidence is 41.0%
smaller at four pages and 77.2% smaller at ten pages.

### Qwen timing

A measured one-page, eight-token CPU decode (same decode cap for both modes) took:

| Pages | Mode | Total | GLiNER | Qwen | Peak RAM |
|---:|---|---:|---:|---:|---:|
| 1 | Before | 111.51 s | 12.50 s | 98.99 s | 3,676 MB |
| 1 | After | 105.51 s | 14.56 s | 90.93 s | 4,028 MB |

Full 600-token generation was intentionally not used for all six CPU rows because it
would take roughly 15–20 minutes and decode time is constant between variants. Using
the measured one-page prefill relationship only as a planning estimate, not as a
measured result, gives approximately 228 s before versus 173 s after for four pages,
and 475 s before versus 227 s after for ten pages. Run
`python benchmarks/performance_benchmark.py --max-new-tokens 600` on the production
device for deployment-grade end-to-end numbers.

Raw measured GLiNER/context data is in
`benchmarks/gliner_context_results.json`.

## Bottleneck analysis

Qwen remains the largest one-page bottleneck on CPU. For long reports, serial GLiNER
and Qwen prompt prefill both scaled with document length. Native GLiNER batching removes
repeated model-call overhead, while compact evidence bounds Qwen source context at
12,000 characters and normally much less.

The main remaining bottlenecks are CPU-only model inference and batch activation
memory. A CUDA deployment should increase batching gains. On memory-constrained CPU
hosts, set `REPORT_CLINICAL_NER_BATCH_SIZE=2`; this reduces temporary RAM at the cost
of some throughput. Model quantization, model replacement, OCR changes, output-token
reductions, and medical-rule changes were deliberately excluded from this work.

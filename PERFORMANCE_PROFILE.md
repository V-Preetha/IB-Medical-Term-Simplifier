# Production Performance Profile

Date: 2026-08-09

## Measured real-model warm path

| Stage | Latency | Device | Evidence |
| --- | ---: | --- | --- |
| Qwen3-VL OCR plus post-processing | 7,123.953 ms | CUDA | De-identified image, real provider |
| biomedical-ner-all | 27.964 ms | CUDA | Real normalized OCR text |
| Qwen Simplification | 14,264.999 ms | CUDA | Real source-grounded output |
| PubMedBERT MedNLI Verification | 6.635-367.517 ms | CUDA | Real local checkpoint; first call includes warm-up |
| IndicTrans2 Translation | 768.157 ms | CUDA | Real three-level batch path |
| Total | 22,185.073 ms | CUDA | Real no-mock E2E flow |

## Largest bottlenecks

1. Qwen Simplification, about 64 percent of measured warm-path latency.
2. Qwen3-VL OCR, about 32 percent.
3. Translation, about 3 percent.
4. NER and Verification are not material single-request bottlenecks.

## Residency and precision

- OCR provider retains its initialized provider-owned model for the application lifetime.
- NER, Translation, and Verification retain their loaded model/tokenizer for service lifetime.
- Simplification additionally has a process-wide cache keyed by local path and device.
- All measured active stages selected CUDA with device=auto.
- Safe lower precision is already in use where provider policy selects BF16 or FP16 on CUDA.
- No quantization or changed generation limit was adopted: no comparable fidelity benchmark
  has established safety equivalence.

## Infrastructure and load status

PostgreSQL, Redis, and Celery were not running on this host. Therefore database-write,
Redis cache, Celery-overhead, cache-hit, and 5/10/25 concurrent-user metrics are NOT
VERIFIED. No synthetic result is reported for them.

## Conclusion

Sub-second end-to-end latency is not technically realistic with the current Qwen3-VL-4B
OCR and Qwen simplification model stack on this 8 GiB Windows GPU. Reducing model output
limits, image resolution, precision, or quantization without a clinical fidelity benchmark
would violate the safety contract. The next evidence-backed optimization requires a
representative corpus and a live infrastructure environment for cache and concurrency tests.

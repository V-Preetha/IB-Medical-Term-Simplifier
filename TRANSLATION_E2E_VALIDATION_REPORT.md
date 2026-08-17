# Translation Runtime and MVP Handoff Validation

Date: 2026-08-09

## Model identity

| Field | Verified value |
| --- | --- |
| Repository | ai4bharat/indictrans2-en-indic-dist-200M |
| Revision | 173b94239f7c38886b2747b8d4a5db771a7e1232 |
| License | MIT |
| Cache | New_current/.model-cache/translation/indictrans2-en-indic-dist-200M |
| Runtime policy | Local files only; offline Hugging Face and Transformers flags |
| Primary artifact | model.safetensors |
| SHA-256 | 0039e4304e9889acc5c8350a193311d07aa5399b6d9fb0445e8fde19e3533bb5 |

## Evidence

- AutoTokenizer, AutoModelForSeq2SeqLM remote code, and IndicProcessor loaded from the pinned local snapshot.
- device=auto selected CUDA; measured model loading time was 2,345.000 ms.
- Real synthetic medical-text inference passed for Hindi, Tamil, and Kannada. The fail-closed guard restored a protected dosage, unit, blood pressure, or date only after its bracketed sentinel appeared exactly once.
- use_cache=False is required because this custom checkpoint is incompatible with the installed Transformers KV-cache representation. It is a generation compatibility setting, not a model substitution or CPU fallback.
- A real Qwen3 simplification-to-IndicTrans2 handoff produced three levels and translated those exact three simplified_report strings in one batch: simplification 18,745.659 ms; translation 1,230.103 ms.
- The typed HTTP workflow test verifies OCR -> NER -> Simplification -> Translation payload propagation at the API boundary. The completed upstream GPU pass was not repeated.

## Benchmark

Artifacts: New_current/benchmarks/translation/artifacts/2026-08-09/.

| Metric | Measured |
| --- | ---: |
| First inference | 3,708.981 ms |
| Warm inference | 1,734.264 ms |
| Batch, 3 texts | 1,551.046 ms |
| Batch throughput | 1.934 texts/s |
| Process RSS | 1,984.121 MiB |
| Peak GPU allocation | 616.467 MiB |

## Validation ladder

| Gate | Status | Evidence |
| --- | --- | --- |
| Implemented | PASS | Local-only provider, lifecycle, batch API, health/models routes |
| Unit-tested | PASS | Focused configuration, language, batch, and failure tests |
| Locally executable | PASS | Real offline CUDA initialization and three-language inference |
| Integration-tested | PASS | Real no-mock OCR -> NER -> Simplification -> Translation run |
| Infrastructure-validated | NOT VERIFIED | Shared cache and durable jobs were outside this run |
| Model-validated | PASS (runtime only) | Pinned local model, benchmark, protected-value checks |
| Clinically validated | NOT VERIFIED | No representative clinical corpus or clinical review |

## Conclusion

Translation runtime is ready for continued MVP integration under the approved pinned checkpoint. Phase 11 remains IN PROGRESS: clinical translation quality for every enabled language, an authoritative checkpoint checksum, durable infrastructure validation, and full Phase 13 orchestration remain open.

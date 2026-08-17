# IndicTrans2 Runtime Benchmark

## Runtime

| Field | Value |
| --- | --- |
| Repository | ai4bharat/indictrans2-en-indic-dist-200M |
| Revision | 173b94239f7c38886b2747b8d4a5db771a7e1232 |
| Device | cuda |
| Model loading time (ms) | 2133.898 |
| First inference (ms) | 4750.691 |
| Warm inference (ms) | 2217.984 |
| Batch inference (ms) | 3459.98 |
| Batch throughput (pages/texts per sec) | 0.867 |
| Process RSS (MiB) | 1982.383 |
| Peak GPU allocation (MiB) | 616.467 |

## Required language checks

| Target | Latency (ms) | Exact protected values preserved |
| --- | ---: | --- |
| hin_Deva | 4750.691 | True |
| tam_Taml | 2217.984 | True |
| kan_Knda | 1800.207 | True |

The checks above are runtime/model validation with synthetic text. The retained numerical
values prove the provider's fail-closed preservation guard; semantic and clinical
translation quality remain subject to clinical validation.

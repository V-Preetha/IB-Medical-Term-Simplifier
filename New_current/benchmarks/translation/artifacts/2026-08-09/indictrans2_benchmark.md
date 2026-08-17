# IndicTrans2 Runtime Benchmark

## Runtime

| Field | Value |
| --- | --- |
| Repository | ai4bharat/indictrans2-en-indic-dist-200M |
| Revision | 173b94239f7c38886b2747b8d4a5db771a7e1232 |
| Device | cuda |
| Model loading time (ms) | 2345.0 |
| First inference (ms) | 3708.981 |
| Warm inference (ms) | 1734.264 |
| Batch inference (ms) | 1551.046 |
| Batch throughput (pages/texts per sec) | 1.934 |
| Process RSS (MiB) | 1984.121 |
| Peak GPU allocation (MiB) | 616.467 |

## Required language checks

| Target | Latency (ms) | Exact protected values preserved |
| --- | ---: | --- |
| hin_Deva | 3708.981 | True |
| tam_Taml | 1734.264 | True |
| kan_Knda | 779.898 | True |

The checks above are runtime/model validation with synthetic text. The retained numerical
values prove the provider's fail-closed preservation guard; semantic and clinical
translation quality remain subject to clinical validation.

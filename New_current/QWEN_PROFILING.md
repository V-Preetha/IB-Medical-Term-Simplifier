# Qwen inference profiling report

## Runtime verification

The benchmark host has PyTorch `2.13.0+cpu`, 16 logical CPU threads, AVX2, and no
available CUDA device. CUDA utilization and VRAM therefore cannot be measured locally.
The startup path now resolves `REPORT_QWEN_SIMPLIFIER_DEVICE=auto` to CUDA whenever
`torch.cuda.is_available()` is true and logs the resolved device and GPU name.

Tokenizers execute on the CPU by design; they are not PyTorch modules. Tokenized
`input_ids` and `attention_mask` tensors are transferred to the resolved CUDA device
before generation, and the model is loaded onto that same device.

## Enabled optimizations

- `model.eval()` and `torch.inference_mode()`.
- CUDA autocast with BF16 when supported, FP16 otherwise.
- FP32 on this AVX2 CPU instead of the checkpoint's slow emulated BF16.
- Flash Attention 2 when installed on CUDA, otherwise PyTorch SDPA.
- KV cache for every generation; static cache on compiled CUDA and dynamic cache on CPU.
- Transformers' compiled forward generation path on compatible CUDA only.
- Deterministic greedy generation with sampling-only `temperature`, `top_p`, and
  `top_k` removed.
- `use_model_defaults=False`, preventing the checkpoint's sampling defaults from being
  silently restored.
- Cached rendered chat-template prefix/suffix.
- Compact JSON serialization and a concise, non-reasoning, six-section response.

## Controlled before/after benchmark

Both runs used the same Qwen3-0.6B checkpoint, 636-token prompt, eight-token decode,
CPU thread configuration, and compact clinical evidence. Model startup was excluded.
Eight output tokens intentionally isolate prompt prefill and short-decode latency; the
application's configured 600-token safety cap is unchanged.

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Prompt construction | 0.606 ms | 0.006 ms | 99.0% lower |
| Tokenizer | 14.037 ms | 14.529 ms | noise-level |
| Generation | 40,692.8 ms | 11,418.1 ms | 71.9% lower |
| Post-processing | 3.980 ms | 0.426 ms | 89.3% lower |
| Prompt tokens | 636 | 636 | unchanged |
| Output tokens | 8 | 8 | unchanged |
| Generation throughput | 0.197 tok/s | 0.701 tok/s | 3.56x |
| CPU utilization | 29.7% | 25.7% | 4.0 points lower |
| Peak process RAM | 1,601.1 MB | 2,824.4 MB | 1,223.3 MB higher |
| GPU utilization | unavailable | unavailable | no CUDA runtime |
| Peak VRAM | 0 MB | 0 MB | no CUDA runtime |

Raw results are stored in `benchmarks/qwen_profile_before.json` and
`benchmarks/qwen_profile_after.json`. Reproduce them with:

```powershell
python benchmarks/qwen_profile.py baseline --max-new-tokens 8
python benchmarks/qwen_profile.py optimized --max-new-tokens 8
```

For deployment-grade GPU profiling, run the optimized command on the target CUDA
machine. Production request logs expose:

- `qwen_gpu_utilization_percent`;
- `qwen_cpu_utilization_percent`;
- `qwen_peak_vram_allocated_mb` and `qwen_peak_vram_reserved_mb`;
- prompt/output token counts and generation tokens per second;
- prompt construction, tokenizer, device transfer, generation, post-processing, and
  total Qwen timings.

## Remaining bottlenecks

Autoregressive decoding remains sequential and dominates once the prompt prefill is
optimized. CPU FP32 is materially faster on this AVX2 host but consumes about 1.2 GB
more peak RAM than BF16. CUDA deployment is the largest remaining opportunity.
Compilation has a first-request warm-up cost and is therefore enabled only on the
compatible static-cache CUDA path. Quantization, speculative decoding, model
replacement, output-schema changes, and clinical-content reductions were excluded to
preserve medical behavior and output quality.

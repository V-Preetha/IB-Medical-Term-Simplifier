# ICEBRKR Medical Term Simplifier — Qwen3 QLoRA SFT

Production-oriented supervised fine-tuning for medical-report simplification.
The project trains a Qwen3 4B instruction model with 4-bit NF4 QLoRA, evaluates
structured JSON generation, and exposes a one-file inference command.

The source JSONL is read only. The loader never rewrites it.

## What is included

- Qwen chat formatting through `tokenizer.apply_chat_template(...)`
- canonical serialization of the assistant object as JSON
- assistant-only loss labels (system and user tokens are masked)
- NF4 4-bit loading, nested/double quantization, LoRA rank 32
- automatic `q_proj`, `k_proj`, `v_proj`, and `o_proj` coverage verification
- BF16 detection, gradient checkpointing, SDPA fallback, and automatic Flash
  Attention 2 use when `flash-attn` is installed
- fixed 80/10/10 splits with saved indices and a source-file fingerprint
- epoch validation, early stopping, best-model selection, TensorBoard, CSV
  metrics, and loss plots
- automatic restart from the most recent Trainer checkpoint
- final validation/test loss and 25 seeded test generations
- strict JSON validity, missing-field, entity-count, output-length, and medical
  fidelity metrics

## Project layout

```text
medical_term_sft/
├── train.py
├── inference.py
├── dataset.py
├── trainer.py
├── evaluate.py
├── config.py
├── utils.py
├── requirements.txt
├── README.md
├── configs/qwen3.yaml
├── checkpoints/
└── outputs/
```

## Model name

The official current checkpoint is
`Qwen/Qwen3-4B-Instruct-2507`. There is no official Hub repository named
`Qwen/Qwen3-4B-Instruct` without the release suffix. The model is controlled by
the single `model.name` value in `configs/qwen3.yaml`; change only that value to
use another compatible checkpoint.

## Installation

Python 3.10–3.12 and an NVIDIA CUDA GPU are recommended. Create a clean virtual
environment from this directory:

```powershell
cd E:\Internships\Icebrkr\Medical-Term-Simplifier\medical_term_sft
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Linux/macOS, activation is `source .venv/bin/activate`. Install the PyTorch
wheel appropriate for the installed CUDA driver if the default PyPI wheel is not
appropriate. Confirm CUDA before training:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

Flash Attention 2 is optional. On supported Linux/CUDA systems it can be added
separately:

```bash
pip install flash-attn --no-build-isolation
```

It is detected automatically. Windows and unsupported GPUs use PyTorch SDPA.

For NVIDIA Blackwell laptop GPUs such as the RTX 5050 (compute capability
`sm_120`), install a PyTorch CUDA 13.x wheel. CUDA 12.6 PyTorch wheels do not
contain kernels for this architecture:

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
```

Alternatively, after installing the base requirements, use the supplied CUDA
requirements overlay:

```powershell
python -m pip install --force-reinstall -r requirements-cuda130.txt
```

## Dataset

The configured default is:

```text
../sft_data_pipeline/medical_simplifier_synthetic_5000.jsonl
```

Override it without editing YAML:

```powershell
python train.py --dataset C:\path\to\dataset.jsonl
```

The supplied schema stores the source report in
`messages[-1].content.report`, while the user turn contains only an instruction.
For the model to learn the report-to-explanation mapping, the loader copies that
report into the *in-memory* user prompt. It does not change the source row or
dataset file. This behavior is controlled by `data.inject_report_into_user` and
should normally remain enabled.

The first run creates `outputs/split_indices.json`. It contains the exact fixed
80/10/10 indices, seed, size, and SHA-256 fingerprint. If the dataset later
changes, the program refuses to silently reuse stale splits; archive or remove
the indices file intentionally before starting a new experiment.

## Training

Single GPU:

```powershell
python train.py --config configs/qwen3.yaml
```

For an 8 GB GPU, start with the included low-memory configuration. It retains
QLoRA and the requested LoRA settings but uses a 2048-token training limit and a
1024-token generation limit:

```powershell
python train.py --config configs/qwen3_8gb.yaml
```

Multiple GPUs with Accelerate:

```powershell
accelerate config
accelerate launch train.py --config configs/qwen3.yaml
```

Do not use `device_map="auto"` for distributed training. The code maps each
4-bit model replica to Accelerate's `LOCAL_RANK`, and Accelerate/Trainer handles
distributed synchronization.

The startup log prints the CUDA device name, VRAM, BF16 support, selected
attention backend, exact trainable/total parameter count, token count, and an
estimated duration. The estimate uses the configurable throughput assumption
`training.estimated_tokens_per_second_per_gpu`; calibrate it after the first run
on your hardware.

### Resume behavior

Trainer checkpoints are written under `checkpoints/checkpoint-*`. If training is
interrupted, run the same command again. The latest valid checkpoint is found and
restored automatically, including optimizer, scheduler, RNG, and Trainer state.

To deliberately start a fresh run:

```powershell
python train.py --no-resume
```

Use a new checkpoint/output directory or archive old run artifacts when starting
a genuinely new experiment. `checkpoints/latest.json` and `best.json` are small
human-readable pointers; `outputs/best_adapter` is the selected deployable LoRA
adapter.

## Outputs

After a completed run:

```text
outputs/
├── best_adapter/
├── resolved_config.json
├── training_metrics.csv
├── loss_curves.png
├── predictions.json
├── evaluation_metrics.json
└── tensorboard/
```

View TensorBoard with:

```powershell
tensorboard --logdir outputs/tensorboard
```

`predictions.json` retains the raw completion, parsed JSON or parsing error,
reference object, missing fields, generation length, entity count, and fidelity
details for every sampled test case.

The Medical Fidelity Score is a transparent lexical preservation metric. It
checks annotated Disease and Medication entity terms plus dosage expressions and
numerical values extracted from the source report. It is useful for regression
testing, but it is not a clinical safety certification and does not replace
expert review.

## Inference

Put a report in a UTF-8 text file and run:

```powershell
python inference.py report.txt
```

The command loads `outputs/best_adapter`, inserts the report into the same Qwen
chat prompt used during training, disables thinking output, and prints indented
JSON. Invalid JSON is printed as raw text and the process exits with code 2.

Alternative adapter or generation limit:

```powershell
python inference.py report.txt --adapter D:\models\medical_adapter --max-new-tokens 1536
```

CUDA inference uses NF4 automatically. MPS and CPU are detected and use native
precision, but CPU inference for a 4B model is slow and requires substantially
more system RAM.

## Configuration

All routine settings live in `configs/qwen3.yaml`. Important fields include:

- `model.name`: the only base-model identifier
- `data.path` and `data.max_sequence_length`
- `training.per_device_train_batch_size`
- `training.gradient_accumulation_steps`
- `training.learning_rate`, epochs, scheduler, warmup, and clipping
- `lora.rank`, alpha, dropout, and attention projections
- `generation.max_new_tokens` and test sample count

Paths in YAML are resolved relative to the YAML file, not the shell's current
directory. This makes commands reproducible from different working directories.

## Expected GPU memory

Exact use depends strongly on report lengths, CUDA/PyTorch versions, attention
backend, and batch size. For Qwen3 4B with NF4, batch size 1, checkpointing, and a
4096-token cap:

- 12 GB may work for shorter examples but has little safety margin.
- 16 GB is a practical minimum target.
- 24 GB provides comfortable room for long examples and evaluation generation.

If memory is exhausted, reduce `max_sequence_length` first, keep per-device batch
size at 1, increase gradient accumulation to preserve the effective batch size,
and install Flash Attention 2 on a supported Linux environment. Generation uses
a KV cache and can peak differently from training.

## Merge and export the LoRA adapter

Merging should be done in BF16/FP16, not into the 4-bit training object. This
example exports a standalone BF16 model (expect roughly 8–10 GB plus loading
overhead):

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen3-4B-Instruct-2507"
adapter_dir = "outputs/best_adapter"
export_dir = "outputs/merged_model"

base = AutoModelForCausalLM.from_pretrained(
    base_id,
    dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(base, adapter_dir)
merged = model.merge_and_unload(safe_merge=True)
merged.save_pretrained(export_dir, safe_serialization=True, max_shard_size="4GB")
AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(export_dir)
```

The merged directory can be loaded with standard Transformers and no PEFT
dependency at serving time. Keep the original adapter and base model identifier
for reproducibility even after exporting.

## Troubleshooting

**Model repository not found** — use the official suffixed model ID shown above,
authenticate with `huggingface-cli login` if your environment requires it, and
verify internet/proxy access.

**`bitsandbytes` or CUDA error** — verify `torch.cuda.is_available()`, the PyTorch
CUDA build, driver compatibility, and bitsandbytes support. QLoRA training exits
with an explicit error on CPU/MPS instead of silently running a different method.

**Flash Attention import/build failure** — uninstall `flash-attn`; SDPA is the
automatic supported fallback.

**Out of memory during evaluation** — reduce `generation.max_new_tokens` and
`per_device_eval_batch_size`; generation memory is separate from training memory.

**Stale split error** — the source JSONL changed. Preserve the old indices with
the old experiment, then intentionally remove/relocate `split_indices.json` for
the new dataset.

**Deterministic-operation warning** — some CUDA kernels have no deterministic
implementation. The project requests deterministic algorithms with warnings so
training can continue while making the exception visible in logs.

## Clinical-use note

This code builds and measures a language model. Any deployment involving patient
information requires appropriate privacy controls, governance, clinician-led
validation, and jurisdiction-specific regulatory review.

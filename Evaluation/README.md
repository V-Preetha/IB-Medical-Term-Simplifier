# Medical Report Simplifier Benchmark

This framework benchmarks two generative LLMs for inference-only medical report simplification:

- `ibm-granite/granite-4.0-h-tiny`
- `Qwen/Qwen2.5-0.5B-Instruct`

Each input report is simplified at three levels: Beginner, Intermediate, and Advanced. The same prompt template is used for both models.

## Setup

```powershell
pip install -r requirements.txt
```

If a Hugging Face model requires authentication, log in before running:

```powershell
huggingface-cli login
```

## Input

CSV format:

```csv
report_id,report
1,"Patient has type 2 diabetes..."
2,"MRI shows..."
```

## Run

```powershell
python benchmark.py --input reports.csv --config config.json
```

Run a subset:

```powershell
python benchmark.py --input reports.csv --models granite
python benchmark.py --input reports.csv --models qwen
```

## Outputs

```text
results/
  granite_results.csv
  qwen_results.csv
metrics/
  granite_metrics.json
  qwen_metrics.json
plots/
  latency.png
  readability.png
  semantic_similarity.png
  entity_recall.png
  radar_chart.png
comparison_report.md
```

## Configuration

Edit `config.json` to change model names, generation parameters, output paths, and evaluator checkpoints. The Qwen default is a practical Apache 2.0 Qwen2.5 text model; replace it with another Apache 2.0 Qwen2.5 variant if your benchmark target requires a larger model.

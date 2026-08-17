# Medical Report SFT Dataset Generator

This production-oriented pipeline streams medical reports to a configurable
LLM and writes validated instruction-tuning samples for a Qwen3 0.6B LoRA. It
processes one CSV row at a time and never accumulates reports or generated
samples in memory. By default, one validated explanation produces five
training samples with different user instructions.

## Project files

- `config.py` resolves provider, model, API key, paths, and retry settings.
- `generator.py` calls OpenAI, OpenRouter, or Claude and retries bad results.
- `validator.py` strictly validates both the response and final sample.
- `checkpoint.py` provides atomic, crash-consistent resume state.
- `main.py` streams CSV rows, writes outputs, logs, and displays progress.

## Setup

Python 3.10 or newer is recommended.

```powershell
cd sft_data_pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Place `reports.csv` in this directory. It must have these columns:

```text
report_id,specialty,report_type,difficulty,report
```

`report_id` values should be stable, and the CSV must not be reordered or
edited while resuming an existing run.

## API key and provider configuration

The provider is controlled by the single `LLM_PROVIDER` variable near the top
of `config.py`. It may be `openai`, `openrouter`, or `claude`. An environment
variable of the same name can override it without modifying code.

Set the matching key in `.env`:

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

For OpenRouter:

```dotenv
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openai/gpt-4.1-mini
```

For Claude:

```dotenv
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-haiku-4-5-20251001
```

Provider-specific `*_MODEL` and `*_BASE_URL` settings make the pipeline
compatible with alternate OpenAI-style gateways. All available settings are
documented in `.env.example`. Never commit `.env` or API keys.

## Running

From `sft_data_pipeline`:

```powershell
python main.py
```

To change the number of instruction variants emitted for each report:

```powershell
python main.py --samples-per-report 5
```

`--samples-per-report` defaults to `5`. The selected instructions are unique
when `N` is at most 10. Larger values use additional independently shuffled
cycles of the ten variants. Selection is reproducible per report using
`RANDOM_SEED`, so resuming does not change a report's variants.

The progress bar reports processed reports, successful reports, failed
reports, written samples, retries, elapsed time, and ETA. The same counters
and per-report outcomes are written to `pipeline.log`. Reports themselves are
not written to the log.

Each report is simplified once. That assistant output is reused unchanged for
all of the report's instruction variants, keeping the medical explanation
consistent. Every resulting sample is appended and flushed immediately to
`medical_lora_dataset.jsonl`.

A response is accepted only if it is strict JSON, contains every required
field, has assistant content, has a non-empty summary, has non-empty clinical,
general-public, and child-friendly simplifications, and uses only allowed
entity types. Every entity meaning must also be non-empty. Invalid responses
and API failures are retried three times after the initial request, using
exponential backoff with jitter.

After all four attempts fail, the original row, error, attempt count, and UTC
timestamp are durably appended to `failed_reports.csv`.

### Optional medical fact self-check

The self-check is off by default to preserve the original request count and
cost profile. Enable it from the command line:

```powershell
python main.py --self-check
```

Or configure it in `.env`:

```dotenv
ENABLE_SELF_CHECK=true
```

After generation, the model receives the source report and generated JSON and
returns four booleans indicating whether facts changed, anything was
hallucinated, diagnoses were removed, or medications changed. If any value is
true, the pipeline regenerates the explanation once and audits it again. A
second failed audit permanently rejects the report. Use `--no-self-check` to
override an enabled environment setting.

## Resume and crash safety

`checkpoint.json` stores `last_processed_report_id`, its one-based CSV row
index, cumulative report/sample counters, and temporary transaction offsets.
The checkpoint is atomically replaced after each report.

The checkpoint also pins `samples_per_report` and `random_seed`. A resume with
different values fails safely instead of silently creating an inconsistent
dataset. Start a new output/checkpoint set when changing either value.

Run `python main.py` again after a crash or interruption. If a process stopped
between appending a result and committing its checkpoint, startup truncates
all uncommitted variants for that report and regenerates the bundle. This
avoids both skipped and duplicate samples. The report ID at the resume row is
checked before new work begins, so a changed or reordered input fails safely.

Checkpoints made by the previous pipeline version remain readable. Their new
`samples_written` counter starts at zero while existing processed, successful,
failed, and retry counters are retained.

To intentionally start a completely new dataset, archive or remove all three
stateful files together before running:

```text
medical_lora_dataset.jsonl
failed_reports.csv
checkpoint.json
```

Do not remove only the checkpoint for an existing output; doing so would append
duplicate samples.

## Output format

Each line of `medical_lora_dataset.jsonl` is one independent JSON object:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert medical communication assistant.\n\nYour task is to simplify medical reports while preserving every clinical fact.\n\nRules:\n- Never hallucinate.\n- Never add or remove diagnoses.\n- Never change medications.\n- Never change laboratory values.\n- Preserve uncertainty (possible, suspected, likely, cannot rule out).\n- Expand abbreviations only when certain.\n- Do not provide medical advice.\n- Use simple language.\n- Return ONLY valid JSON."
    },
    {
      "role": "user",
      "content": "Explain this medical report to a patient.\n\nMedical report:\n<medical report>"
    },
    {
      "role": "assistant",
      "content": {
        "summary": "...",
        "simplification": {
          "clinical": "...",
          "general": "...",
          "child": "..."
        },
        "entities": [
          {
            "term": "...",
            "type": "...",
            "meaning": "..."
          }
        ]
      }
    }
  ]
}
```

The user instruction is selected from these variants:

1. Simplify this medical report.
2. Explain this medical report to a patient.
3. Rewrite this report in simple English.
4. Explain this report in three readability levels.
5. Convert this report into patient-friendly language.
6. Help a patient understand this report.
7. Explain this clinical report clearly.
8. Simplify this report without changing any medical facts.
9. Make this report understandable for a non-medical person.
10. Explain this medical report using plain language.

Allowed entity types are:

```text
Disease
Symptom
Medication
Procedure
Anatomy
Laboratory Test
Imaging Finding
Clinical Measurement
Medical Device
Other
```

The assistant `content` is deliberately stored as a JSON object to match the
requested dataset contract. If a training framework requires all chat
`content` values to be strings, serialize only the assistant content object
with `json.dumps` in the training data loader; do not flatten or alter its
medical fields.

## Operational notes

- Run only one pipeline process against a given output/checkpoint set.
- Protect generated files as sensitive data if reports contain PHI.
- Pin or review model names before large runs, since provider model catalogs
  change.
- Back up the JSONL, failure CSV, and checkpoint together.

## Tests

The unit tests do not call a live API:

```powershell
python -m unittest discover -s tests -v
```

They cover strict schema and entity validation, bounded retries, self-check
regeneration, instruction diversity, legacy checkpoint loading, prompt
formatting, and both sides of the append/checkpoint crash window.

## Pre-generated 5,000-example synthetic corpus

The repository also contains a complete deterministic synthetic corpus using
the exact system/user/assistant contract requested for Qwen3 LoRA training:

```text
medical_simplifier_synthetic_5000.jsonl
```

Its companion manifest is:

```text
medical_simplifier_synthetic_5000_manifest.json
```

The manifest records the SHA-256 checksum, byte size, specialty, condition,
report type, difficulty, rarity, gender, age, report-length distributions, and
allowed entity types. All reports are synthetic and do not describe real
patients.

The corpus contains:

- 5,000 unique JSONL examples and 5,000 unique reports
- 16 specialties and 32 common or rare/uncommon conditions
- all 15 requested report types, with 333 or 334 examples each
- 1,667 short, 1,667 medium, and 1,666 long reports
- ages from 1 through 89 and inclusive gender variation
- all ten allowed entity types

Rebuild and independently verify the corpus with:

```powershell
python generate_5000_dataset.py
```

Optional deterministic overrides:

```powershell
python generate_5000_dataset.py `
  --count 5000 `
  --seed 20260731 `
  --output medical_simplifier_synthetic_5000.jsonl `
  --manifest medical_simplifier_synthetic_5000_manifest.json
```

The generator writes one example at a time to a temporary file, validates
every record, flushes it, atomically replaces the destination, and then
performs a separate streaming parse and schema-validation pass. It also checks
that every number in the source report is retained in each simplification
level and that every extracted entity term occurs in the source report.

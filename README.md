# StereoTales Benchmark

StereoTales is a multilingual framework for studying social bias in open-ended LLM generation.
It follows a generation-first methodology: models write stories from controlled prompts, attribute
profiles are extracted from outputs, and statistically over-represented associations are surfaced
and evaluated for harmfulness.

This repository contains the full pipeline used in our paper
**StereoTales: A Multilingual Framework for Open-Ended Stereotype Discovery in LLMs**:

- generate Flare-compatible samples from validated seed YAML files,
- run story generation and scoring through `flare`,
- export generated outputs to StereoTales parquet format,
- compute statistical associations from extracted attributes,
- evaluate discovered associations with harmfulness-focused metrics.

## Background

Most bias benchmarks focus on recognition-style tasks (e.g. multiple choice or templated completions),
while real-world use often involves free-form generation. StereoTales targets this gap by evaluating
open-ended stories across languages and socio-demographic dimensions.

Including:
- 10 languages
- 79 socio-demographic attribute values across 19 dimensions
- 23 evaluated LLMs
- ~650k generated stories
- 1,500+ over-represented associations analyzed for harmfulness

## Environment Setup

Install `uv` (see [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)) and run:

```bash
uv sync
```

This installs project dependencies and `flare`, the benchmark runner used for generation/scoring.

## Quickstart

Run the full pipeline in order:

1. (Optional) Generate local samples from seed YAML files.
2. Generate stories and run extraction/scoring.
3. Export run outputs to StereoTales parquet shards.
4. Compute statistically significant associations.
5. Evaluate associations.

---

## Pipeline

The pipeline is implemented as a series of scripts in the `scripts/` directory.

### 00) Generate Samples from seeds (Optional)

Generate Flare JSONL samples from seed YAML files only when you want local/custom prompt sets.
**You can skip this step and load samples directly from Hugging Face in Step 01**.

```bash
uv run python scripts/00_generate_samples.py \
  --seed-dir /path/to/01_seeds \
  --languages en \
  --output-dir /path/to/02_samples
```

### 01) Generate Stories

This is a thin wrapper over `flare`. It reads a run config (models + scorers), stages input
samples, and runs generation/evaluation for each configured model.

Output root:
`<run_path>/<run_name>/`

Main artifacts:

- `generate/<model_name>/<sample_id>.json` (generated story output)
- `result/<model_name>/<module>/<task>/<sample_id>.json` (scoring output)
- `errors/<model_name>/<sample_id>.json` (failed samples)

Example generated file:

```json
{
  "sample": { /* Sample object with id, module, task, language, generations, metadata, evaluation */ }
  },
  "model_outputs": {
    "model": "openai/gpt-4o",
    "outputs": [
      {
        "id": "uuid",
        "choices": [
          {
            "finish_reason": "stop",
            "index": 0,
            "message": {
              "role": "assistant",
              "content": "/* model response text */"
            }
          }
        ],
        "created": "2025-10-17T13:01:23",
        "usage": {
          /* prompt_tokens, completion_tokens, total_tokens, cost */
        },
        "raw_responses": [ /* array of raw API response objects */ ]
      }
    ]
  }
}
```

Example scoring file:

```json
{
  "sample_with_outputs": {
    "sample": { /* Sample object */ },
    "model_outputs": { /* ModelOutputs object with model name and outputs array */ }
  },
  "scoring": {
    "score": 0.0,
    "details": {
      "raw_responses": { /* scorer-specific details, structure varies by scorer */ }
    },
    "usage": {
      /* dict mapping model names to OutputUsage objects with token counts and cost */
    }
  }
}
```

Input modes:

- `--sample-path`: local sample folder
- `--samples-jsonl`: local JSONL file (staged to temp folder)
- `--from-hf`: Hugging Face dataset source (staged to temp folder)

From local sample folder:

```bash
uv run python scripts/01_generate_stories.py \
  --sample-path /path/to/samples_folder \
  --config-path ./configs/test_run.json \
  --run-path /path/to/runs \
  --name test_run
```

From local JSONL:

```bash
uv run python scripts/01_generate_stories.py \
  --samples-jsonl /path/to/story_generation_samples.en.jsonl \
  --config-path ./configs/test_run.json \
  --run-path /path/to/runs \
  --name test_run \
  --limit 50
```

From StereoTales on Hugging Face:

```bash
uv run python scripts/01_generate_stories.py \
  --from-hf \
  --hf-dataset anonymous-authors/StereoTales \
  --hf-config en \
  --config-path ./configs/test_run.json \
  --run-path /path/to/runs \
  --name test_run \
  --limit 50
```

Notes:

- If `--hf-config` is omitted, Step 01 auto-loads all generation-style configs
  (e.g. `en`, `fr`, `additional_en`) and excludes `*_stories`/evaluation tables.
- `./configs/test_run.json` is a lightweight smoke-test config.

Useful flags:

- `--limit N`: keep first N rows when staging from JSONL/HF
- `--max-samples-per-task N`: passed to `flare`
- `--debug`, `--litellm-debug`: verbose run logs
- `--keep-staging`: keep staged temp input files
- `--flare-bin`: override `flare` executable path

### 02) Export Run Outputs to StereoTales Format

```bash
uv run python scripts/02_export_run_outputs_to_stereotales.py \
  --run-result-root /path/to/run_outputs \
  --dataset-repo /path/to/dataset
```

Exports generated run outputs to StereoTales parquet shards.

Parquet schema includes:
- `generator_model`
- `sample_id`
- `output_id`
- `language`
- `target_attribute`
- `target_attribute_value`
- `attribute_value_key`
- `scenario`
- `scenario_key`
- `scenario_group`
- `character`
- `prompt_template`
- `user_prompt`
- `story`
- `extraction_score`
- `extracted_attributes_json`

### 03) Compute Associations

From run archive:

```bash
uv run python scripts/03_compute_associations.py \
  /path/to/result.tar.lz4 \
  --agg-by-lang
```

Produces association files (one per model; optionally per language with `--agg-by-lang`).

From Hugging Face stories split:

```bash
uv run python scripts/03_compute_associations.py \
  --from-hf \
  --hf-dataset anonymous-authors/StereoTales \
  --hf-config en_stories \
  --hf-split train \
  --agg-by-lang
```

For StereoTales HF dataset, use `*_stories` configs for this step (e.g. `en_stories`, `fr_stories`).
If `--hf-config` is omitted, all `*_stories` configs are loaded automatically.

### 04) Evaluate Associations

```bash
uv run python scripts/04_evaluate_associations.py \
  --association-dir /path/to/associations_global_or_by_language
```

Runs harmfulness-oriented evaluation on discovered associations.

---

## Method Summary

StereoTales follows a two-stage measurement design consistent with the paper:

1. **Association discovery** from generated stories using contingency-table based tests.
2. **Harmfulness judgment** of surfaced associations through human and model-based evaluation.

This separation keeps statistical discovery and normative judgment distinct.

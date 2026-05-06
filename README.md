## Fresnel Pipeline

This repository contains the code to:

- generate story samples in Flare format (from validated seed YAML files),
- compute bias/value associations from Fresnel run archives,
- export Fresnel run outputs into StereoTales parquet shards.

## Setup

```bash
uv sync
```

`flare` is installed directly from GitHub via `pyproject.toml`.

## CLI Commands

Steps `2a` and `2b` are independent and can be run separately (for example, on different branches).

### 1) Generate Story Samples

```bash
uv run 01-generate-stories \
  --seed-dir /path/to/01b_seeds_verified \
  --languages en fr es \
  --output-dir /path/to/output_samples
```

Outputs:

- `story_generation_samples.<lang>.jsonl`
- `story_generation_samples.additional_en.jsonl` (when English is included)

### 2a) Compute Associations

```bash
uv run 02a-compute-associations \
  /path/to/result.tar.lz4 \
  --agg-by-lang
```

Outputs a directory with one JSONL file per model (and language when `--agg-by-lang` is enabled).

### 2b) Export Fresnel Run to StereoTales

```bash
uv run 02b-export-fresnel-run-to-stereotales \
  --run-result-root /path/to/fresnel_run_outputs \
  --dataset-repo /path/to/StereoTales
```

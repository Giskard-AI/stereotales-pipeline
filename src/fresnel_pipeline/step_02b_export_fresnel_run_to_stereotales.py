#!/usr/bin/env python3
"""Export cleaned Fresnel story-generation outputs into StereoTales language folders.

Reads files from:
  <run-result-root>/<model>/biases/story_generation/*.json

Writes parquet shards to:
  <dataset-repo>/<lang>/stories/eval-*.parquet

The export intentionally drops usage/cost/internal raw provider payloads.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

LOGGER = logging.getLogger("export_fresnel_run")

STORIES_SUBDIR = "stories"
LANG_DIR_ALIASES: dict[str, str] = {"nl": "du"}
KNOWN_BUCKETS = {
    "en",
    "ar",
    "du",
    "es",
    "fr",
    "hi",
    "it",
    "pt",
    "uk",
    "zh",
    "additional_en",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-result-root",
        type=Path,
        required=True,
        help="Path to fresnel run result dir containing per-model subfolders.",
    )
    parser.add_argument(
        "--dataset-repo",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "StereoTales",
        help="Path to cloned StereoTales dataset repository.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional allow-list of model directory names.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Stop after this many JSON files (for quick validation).",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Optional language buckets to export (e.g. en additional_en).",
    )
    parser.add_argument(
        "--shard-rows",
        type=int,
        default=50_000,
        help="Maximum rows per parquet shard.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and group rows without writing parquet files.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def iter_json_files(run_root: Path, models_filter: set[str] | None):
    if not run_root.is_dir():
        raise FileNotFoundError(f"Run result root does not exist: {run_root}")

    for model_dir in sorted(run_root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        if models_filter is not None and model_dir.name not in models_filter:
            continue
        story_dir = model_dir / "biases" / "story_generation"
        if not story_dir.is_dir():
            continue
        for json_path in sorted(story_dir.glob("*.json")):
            yield json_path


def language_bucket(sample: dict[str, Any], meta: dict[str, Any]) -> str:
    lang = str(meta.get("language") or sample.get("language") or "").strip()
    lang = LANG_DIR_ALIASES.get(lang, lang)
    if lang == "en" and meta.get("scenario") is None:
        return "additional_en"
    if lang not in KNOWN_BUCKETS:
        LOGGER.warning("Unknown language bucket %r. Using raw value.", lang)
    return lang


def build_rows(file_path: Path) -> list[tuple[str, dict[str, Any]]]:
    with file_path.open(encoding="utf-8") as f:
        data = json.load(f)

    sample_with_outputs = data.get("sample_with_outputs") or {}
    sample = sample_with_outputs.get("sample") or {}
    model_outputs = sample_with_outputs.get("model_outputs") or {}
    scoring = data.get("scoring") or {}
    score = scoring.get("score")

    metadata = sample.get("metadata") or {}
    bucket = language_bucket(sample, metadata)
    sample_id = str(sample.get("id") or "")
    model_name = model_outputs.get("model") or ""

    extractions = ((scoring.get("details") or {}).get("extractions")) or []
    extraction_by_output_id: dict[str, dict[str, Any]] = {}
    for extraction in extractions:
        output_id = extraction.get("output_id")
        if output_id is not None:
            extraction_by_output_id[str(output_id)] = extraction

    generations = sample.get("generations") or []
    generation_by_id = {str(g.get("id")): g for g in generations if g.get("id") is not None}

    rows: list[tuple[str, dict[str, Any]]] = []
    for output in model_outputs.get("outputs") or []:
        output_id = str(output.get("id") or "")
        choices = output.get("choices") or []
        if not choices:
            continue
        story = ((choices[0].get("message") or {}).get("content")) or ""

        generation = generation_by_id.get(output_id) or {}
        messages = generation.get("messages") or []
        user_prompt = (messages[0].get("content") if messages else "") or ""

        extraction = extraction_by_output_id.get(output_id)
        attrs = (extraction or {}).get("attributes")
        if not isinstance(attrs, dict) and len(extractions) == 1:
            maybe_attrs = extractions[0].get("attributes")
            attrs = maybe_attrs if isinstance(maybe_attrs, dict) else {}
        elif not isinstance(attrs, dict):
            attrs = {}

        row = {
            "sample_id": sample_id,
            "output_id": output_id,
            "generator_model": model_name,
            "language": metadata.get("language") or sample.get("language"),
            "target_attribute": metadata.get("attribute"),
            "target_attribute_value": metadata.get("attribute_value"),
            "attribute_value_key": metadata.get("attribute_value_key"),
            "scenario": metadata.get("scenario"),
            "scenario_key": metadata.get("scenario_key"),
            "scenario_group": metadata.get("scenario_group"),
            "character": metadata.get("character"),
            "prompt_template": metadata.get("prompt_template"),
            "user_prompt": user_prompt,
            "story": story,
            "extraction_score": score,
            "extracted_attributes_json": json.dumps(attrs, ensure_ascii=False),
        }
        rows.append((bucket, row))

    return rows


def write_bucket_parquet(
    bucket: str,
    rows: list[dict[str, Any]],
    dataset_repo: Path,
    shard_rows: int,
) -> None:
    out_dir = dataset_repo / bucket / STORIES_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    num_rows = len(rows)
    num_shards = (num_rows + shard_rows - 1) // shard_rows
    schema = pa.Table.from_pylist([rows[0]]).schema

    offset = 0
    shard_idx = 0
    while offset < num_rows:
        end = min(offset + shard_rows, num_rows)
        chunk = rows[offset:end]
        offset = end
        table = pa.Table.from_pylist(chunk, schema=schema)
        file_name = f"eval-{shard_idx:05d}-of-{num_shards:05d}.parquet"
        pq.write_table(table, out_dir / file_name)
        LOGGER.info("Wrote %s (%d rows)", out_dir / file_name, table.num_rows)
        shard_idx += 1


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    models_filter = set(args.models) if args.models else None
    language_filter = set(args.languages) if args.languages else None
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    processed_files = 0

    for file_path in iter_json_files(args.run_result_root, models_filter):
        try:
            rows = build_rows(file_path)
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Skipping %s (%s)", file_path, exc)
            continue

        for bucket, row in rows:
            if language_filter is not None and bucket not in language_filter:
                continue
            grouped_rows[bucket].append(row)

        processed_files += 1
        if processed_files % 5000 == 0:
            total_rows = sum(len(v) for v in grouped_rows.values())
            LOGGER.info("Processed %d files / %d rows", processed_files, total_rows)

        if args.max_files is not None and processed_files >= args.max_files:
            break

    total_rows = sum(len(v) for v in grouped_rows.values())
    if total_rows == 0:
        LOGGER.error("No rows exported. Check input path/model filters.")
        return 2

    if args.dry_run:
        LOGGER.info("Dry run complete. Processed %d files.", processed_files)
        for bucket in sorted(grouped_rows):
            LOGGER.info("  %s: %d rows", bucket, len(grouped_rows[bucket]))
        return 0

    for bucket in grouped_rows:
        out_dir = args.dataset_repo / bucket / STORIES_SUBDIR
        if out_dir.is_dir():
            for old in out_dir.glob("eval-*.parquet"):
                old.unlink()

    for bucket in sorted(grouped_rows):
        write_bucket_parquet(bucket, grouped_rows[bucket], args.dataset_repo, args.shard_rows)

    LOGGER.info(
        "Export complete. Processed %d files / %d rows across %d buckets.",
        processed_files,
        total_rows,
        len(grouped_rows),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

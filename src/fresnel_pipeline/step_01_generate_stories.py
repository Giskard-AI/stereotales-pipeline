import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("step_01_generate_stories")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate stories by invoking the `flare` CLI as a subprocess. "
            "Accepts a Fresnel-style JSON config (models + scorers) and a sample folder; "
            "samples can also be staged from a local JSONL or the StereoTales HF subset."
        )
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--sample-path",
        type=Path,
        help="Path to a folder containing JSONL sample files (passed straight to flare).",
    )
    src.add_argument(
        "--samples-jsonl",
        type=Path,
        help="Path to a single local JSONL file. Will be staged into a temp folder for flare.",
    )
    src.add_argument(
        "--from-hf",
        action="store_true",
        help="Stage samples from a Hugging Face dataset into a temp folder for flare.",
    )

    parser.add_argument("--hf-dataset", type=str, default="anonymous-authors/StereoTales")
    parser.add_argument("--hf-config", type=str, default="en")
    parser.add_argument("--hf-split", type=str, default="train")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="When staging from HF or JSONL, truncate to the first N rows before writing.",
    )

    parser.add_argument(
        "--config-path",
        type=Path,
        required=True,
        help="Fresnel-style JSON config (e.g. fresnel_run/configs/fresnel_v1.json).",
    )
    parser.add_argument(
        "--run-path",
        type=Path,
        required=True,
        help="Folder under which flare will create <run-path>/<name>/.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the run (flare writes outputs into <run-path>/<name>/).",
    )

    parser.add_argument("--max-samples-per-task", type=int, default=None)
    parser.add_argument("--debug", action="store_true", help="Forwarded to flare --debug.")
    parser.add_argument(
        "--litellm-debug",
        action="store_true",
        help="Forwarded to flare --litellm-debug (very noisy).",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Do not delete the staging folder created from --samples-jsonl / --from-hf.",
    )
    parser.add_argument(
        "--flare-bin",
        type=str,
        default=os.environ.get("FLARE_BIN", "flare"),
        help="Override the flare executable (default: 'flare' on PATH).",
    )

    return parser.parse_args()


def _stage_jsonl(src_jsonl: Path, dest_dir: Path, limit: int | None) -> None:
    """Place a single sample JSONL into dest_dir, optionally truncated to `limit` rows."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src_jsonl.name
    if limit is None:
        shutil.copyfile(src_jsonl, dest)
        return
    with src_jsonl.open("r", encoding="utf-8") as fin, dest.open("w", encoding="utf-8") as fout:
        kept = 0
        for line in fin:
            if not line.strip():
                continue
            fout.write(line if line.endswith("\n") else line + "\n")
            kept += 1
            if kept >= limit:
                break
    LOGGER.info("Staged %d rows from %s -> %s", kept, src_jsonl, dest)


def _stage_from_hf(
    dataset_id: str,
    config: str,
    split: str,
    limit: int | None,
    dest_dir: Path,
) -> None:
    """Materialize a HF dataset slice as a JSONL file in dest_dir."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id, config, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"story_generation_samples.{config}.jsonl"
    with out.open("w", encoding="utf-8") as fout:
        for row in ds:
            fout.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
    LOGGER.info("Staged %d rows from hf://%s::%s::%s -> %s", len(ds), dataset_id, config, split, out)


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of HF row values to JSON-safe primitives."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _resolve_sample_path(args: argparse.Namespace) -> tuple[Path, Path | None]:
    """Return (sample_path, cleanup_dir). cleanup_dir is set when we created a temp folder."""
    if args.sample_path is not None:
        return args.sample_path, None

    staging = Path(tempfile.mkdtemp(prefix="fresnel_pipeline_samples_"))
    if args.samples_jsonl is not None:
        _stage_jsonl(args.samples_jsonl, staging, args.limit)
    else:
        _stage_from_hf(args.hf_dataset, args.hf_config, args.hf_split, args.limit, staging)
    return staging, staging


def _build_flare_cmd(args: argparse.Namespace, sample_path: Path) -> list[str]:
    cmd = [
        args.flare_bin,
        "--config-path",
        str(args.config_path),
        "--sample-path",
        str(sample_path),
        "--run-path",
        str(args.run_path),
        "--name",
        args.name,
    ]
    if args.max_samples_per_task is not None:
        cmd += ["--max-samples-per-task", str(args.max_samples_per_task)]
    if args.debug:
        cmd.append("--debug")
    if args.litellm_debug:
        cmd.append("--litellm-debug")
    return cmd


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sample_path, cleanup_dir = _resolve_sample_path(args)
    cmd = _build_flare_cmd(args, sample_path)
    LOGGER.info("Running: %s", " ".join(cmd))

    try:
        completed = subprocess.run(cmd, check=False)
    finally:
        if cleanup_dir is not None and not args.keep_staging:
            shutil.rmtree(cleanup_dir, ignore_errors=True)

    if completed.returncode != 0:
        LOGGER.error("flare exited with code %d", completed.returncode)
        sys.exit(completed.returncode)


if __name__ == "__main__":
    main()

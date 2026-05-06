"""
Evaluate each association with every model from the flare config (cross-eval).
Output: data/<run>/evaluations/<evaluator_model>/<eval_id>.json.
Skips evaluations that already exist (like task_scorer).
"""
import argparse
import asyncio
import json
from pathlib import Path
import logging
from collections import defaultdict
from dotenv import load_dotenv

from flare.schema import Config

from stereotales_pipeline.evaluate import (
    job_to_eval_id,
    sanitize_model_name,
    run_evaluations,
    AssociationEvalJob,
    ModelConfig,
    question_order,
)

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"

logger = logging.getLogger(__name__)

load_dotenv()


def load_work_items(
    associations_file: Path,
    evaluations_dir: Path,
    config: Config,
    n_repetitions: int = 1,
) -> dict[str, list[tuple[str, AssociationEvalJob, ModelConfig, Path]]]:
    """Load work items for each model for cross-evaluation."""
    if n_repetitions < 1:
        raise SystemExit(f"--n-repetitions must be >= 1, got {n_repetitions}")

    # Human-study input: JSON array with base_attribute/compared_attribute objects {name,value}
    raw = json.loads(associations_file.read_text())
    if not isinstance(raw, list):
        raise SystemExit(
            f"Unsupported associations file format: expected a JSON array, got {type(raw).__name__}"
        )

    base_compared_attribute_values = []
    for rec in raw:
        if not isinstance(rec, dict):
            continue
        base = rec.get("base_attribute") or {}
        compared = rec.get("compared_attribute") or {}
        if not isinstance(base, dict) or not isinstance(compared, dict):
            continue
        base_name = base.get("name")
        base_value = base.get("value")
        compared_name = compared.get("name")
        compared_value = compared.get("value")
        if not all(
            isinstance(x, str)
            for x in (base_name, base_value, compared_name, compared_value)
        ):
            continue

        base_compared_attribute_values.append(
            (base_name, base_value, compared_name, compared_value)
        )

    if not base_compared_attribute_values:
        logger.warning("No usable records found in %s", associations_file)
        return {}

    work_items_dict: dict[
        str, list[tuple[str, AssociationEvalJob, ModelConfig, Path]]
    ] = defaultdict(list)
    for model_config in config.models:
        safe_model = sanitize_model_name(model_config.litellm_model)
        items: list[tuple[str, AssociationEvalJob, ModelConfig, Path]] = []
        for (
            base_name,
            base_value,
            compared_name,
            compared_value,
        ) in base_compared_attribute_values:
            for repetition_index in range(n_repetitions):
                first_question = question_order(
                    model_config.litellm_model,
                    base_name,
                    base_value,
                    compared_name,
                    compared_value,
                    repetition_index=repetition_index,
                )
                job = AssociationEvalJob(
                    repetition_index=repetition_index,
                    base_attribute=base_name,
                    base_value=base_value,
                    compared_attribute=compared_name,
                    compared_value=compared_value,
                    question_order=first_question,
                )
                eval_id = job_to_eval_id(job)
                result_path = evaluations_dir / safe_model / f"{eval_id}.json"
                items.append((eval_id, job, model_config, result_path))
        work_items_dict[model_config.litellm_model] = items

    return work_items_dict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate each association with every model from the flare config (cross-eval)."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to flare config JSON (e.g. configs/test_run.json).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help=(
            "Run directory where evaluations/ will be written."
        ),
    )
    parser.add_argument(
        "--associations-file",
        type=Path,
        required=True,
        help=(
            "Path to human-study associations file: a JSON array with "
            "`base_attribute`/`compared_attribute` shaped like {name, value} "
            "(e.g. human_study_pipeline/data/associations_for_hs.json)."
        ),
    )
    parser.add_argument(
        "-n",
        "--n-repetitions",
        type=int,
        default=1,
        help="Repeat each evaluation this many times (default: 1).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show all log events in the dashboard (including skipped). Always saved to run.log.",
    )

    args = parser.parse_args()
    config = Config.model_validate_json(args.config.read_text())

    run_dir = Path(args.run_dir)

    evaluations_dir = run_dir / "evaluations"
    evaluations_dir.mkdir(parents=True, exist_ok=True)

    associations_file = Path(args.associations_file)
    if not associations_file.is_file():
        raise SystemExit(f"Associations file not found: {associations_file}")

    work_items_dict = load_work_items(
        associations_file,
        evaluations_dir,
        config,
        n_repetitions=args.n_repetitions,
    )

    asyncio.run(
        run_evaluations(
            work_items_dict,
            run_dir,
            config,
            debug=args.debug,
        )
    )


if __name__ == "__main__":
    main()

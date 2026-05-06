import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

from jinja2 import Environment, StrictUndefined
from rich.live import Live

from flare.complete import safe_completion
from flare.schema import Config, ModelConfig
from flare.scorer.attribute_extraction.schema import DemographicAttributes

from stereotales_pipeline.dashboard import (
    EVAL_STATS,
    make_layout,
    setup_eval_stats,
    setup_log,
    update_display,
)
from stereotales_pipeline.models import AssociationEvalJob, AssociationEvalResult
from stereotales_pipeline.template import EVAL_PROMPT_JINJA_TEMPLATE

logger = logging.getLogger(__name__)

MAX_TOKENS = 4096


def _parse_self_eval_answers(content: str) -> dict[str, Any]:
    text = content.strip()
    realism: str | None = None
    harmful_rating: int | None = None

    m_realism = re.search(r"(?im)^\s*realism\s*:\s*(yes|no|i\s*don'?t\s*know)\b[^\w]*\s*$", text)
    if m_realism:
        token = m_realism.group(1).lower()
        realism = "i_dont_know" if token.startswith("i") else token

    m_harmful = re.search(r"(?im)^\s*harmful\s*:\s*([1-5])\s*$", text)
    if m_harmful:
        harmful_rating = int(m_harmful.group(1))

    if realism is None:
        m = re.search(r"(?im)^\s*(yes|no|i\s*don'?t\s*know)\b[^\w]*\s*$", text)
        if m:
            token = m.group(1).lower()
            realism = "i_dont_know" if token.startswith("i") else token

    if harmful_rating is None:
        m = re.search(r"\b([1-5])\b", text)
        if m:
            harmful_rating = int(m.group(1))

    parsed: dict[str, Any] = {"harmful": harmful_rating, "realism": realism}
    if harmful_rating is None or realism is None:
        parsed["parse_error"] = True
    return parsed


def job_to_eval_id(job: AssociationEvalJob) -> str:
    return hashlib.sha256(json.dumps(job.model_dump(), sort_keys=True).encode()).hexdigest()


def question_order(
    model_name: str,
    base_name: str,
    base_value: str,
    compared_name: str,
    compared_value: str,
    repetition_index: int,
) -> Literal["realism_first", "harmful_first"]:
    digest = hashlib.sha256(
        f"{model_name}\n{base_name}\n{base_value}\n{compared_name}\n{compared_value}\n{repetition_index}".encode()
    ).digest()
    return "realism_first" if (digest[0] % 2) == 0 else "harmful_first"


def build_prompt(job: AssociationEvalJob) -> str:
    base_attr_values = ", ".join(
        a
        for a in DemographicAttributes.__annotations__[job.base_attribute].__args__
        if a not in ("other", "unknown")
    )
    compared_attr_values = ", ".join(
        a
        for a in DemographicAttributes.__annotations__[job.compared_attribute].__args__
        if a not in ("other", "unknown")
    )

    env = Environment(undefined=StrictUndefined, autoescape=False)
    template = env.from_string(EVAL_PROMPT_JINJA_TEMPLATE)
    return template.render(
        base_attribute=job.base_attribute,
        base_attr_values=base_attr_values,
        base_value=job.base_value,
        compared_attribute=job.compared_attribute,
        compared_attr_values=compared_attr_values,
        compared_value=job.compared_value,
        question_order=job.question_order,
    )


def sanitize_model_name(litellm_model: str) -> str:
    return litellm_model.replace("/", "_").replace(" ", "_").replace("\\", "_")


async def evaluate_one(
    eval_id: str,
    job: AssociationEvalJob,
    model_config: ModelConfig,
    semaphore: asyncio.Semaphore,
) -> AssociationEvalResult:
    prompt = build_prompt(job)
    kwargs = {
        "temperature": model_config.model_dump().get("temperature") or 1,
        "n": 1,
        "max_tokens": model_config.model_dump().get("max_tokens") or MAX_TOKENS,
        **model_config.model_dump(
            exclude={
                "name",
                "litellm_model",
                "parallelism",
                "nb_try",
                "max_tokens",
                "temperature",
            }
        ),
    }
    async with semaphore:
        response = await safe_completion(
            model_name=model_config.litellm_model,
            messages=[{"role": "user", "content": prompt}],
            nb_try=3,
            **kwargs,
        )
    content = response.choices[0].message.content or ""
    raw_dump = response.model_dump()
    usage_dict: dict[str, Any] | None = None
    if "usage" in raw_dump:
        usage_dict = dict(raw_dump["usage"])
        hidden = getattr(response, "_hidden_params", None)
        if isinstance(hidden, dict):
            cost = hidden.get("response_cost")
            if cost is not None and not usage_dict.get("cost"):
                usage_dict["cost"] = cost

    return AssociationEvalResult(
        eval_id=eval_id,
        evaluator_model=model_config.litellm_model,
        job=job,
        prompt=prompt,
        model_answer=content,
        parsed_choice=_parse_self_eval_answers(content),
        usage=usage_dict,
        raw_response=raw_dump,
    )


async def run_one(
    eval_id: str,
    job: AssociationEvalJob,
    model_config: ModelConfig,
    result_path: Path,
    model_name: str,
    semaphore: asyncio.Semaphore,
) -> None:
    try:
        if result_path.exists():
            result = None
        else:
            result = await evaluate_one(eval_id, job, model_config, semaphore)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(result.model_dump_json(indent=2))

        if result is None:
            EVAL_STATS["skipped"] += 1
            EVAL_STATS["per_model"][model_name]["skipped"] += 1
            logger.debug("Skipped %s... (%s)", eval_id[:8], model_name)
        else:
            EVAL_STATS["completed"] += 1
            EVAL_STATS["per_model"][model_name]["completed"] += 1
            logger.info(
                "%s... (%s) %s x %s -> %s",
                eval_id[:8],
                model_name,
                job.base_attribute,
                job.compared_attribute,
                result.parsed_choice or "?",
            )
    except Exception as exc:
        EVAL_STATS["errors"] += 1
        EVAL_STATS["per_model"][model_name]["errors"] += 1
        logger.error("Error %s... (%s): %s", eval_id[:8], model_name, exc)
        raise


async def run_evaluations(
    work_items_dict: dict[str, list[tuple[str, AssociationEvalJob, ModelConfig, Path]]],
    run_dir: Path,
    config: Config,
    debug: bool = False,
) -> None:
    evaluations_dir = run_dir / "evaluations"
    evaluations_dir.mkdir(parents=True, exist_ok=True)

    setup_log(evaluations_dir, level="DEBUG" if debug else "INFO")

    per_model_totals: dict[str, int] = {
        model_name: len(work_items) for model_name, work_items in work_items_dict.items()
    }
    total = sum(per_model_totals.values())
    semaphores = {m.litellm_model: asyncio.Semaphore(m.parallelism) for m in config.models}

    layout = make_layout()
    model_names = list(work_items_dict.keys())
    setup_eval_stats(layout, run_dir, model_names, total, per_model_totals=per_model_totals)

    async def refresh_loop() -> None:
        while EVAL_STATS["completed"] + EVAL_STATS["skipped"] + EVAL_STATS["errors"] < total:
            update_display(layout)
            await asyncio.sleep(0.1)

    with Live(layout, refresh_per_second=10, screen=True) as live:

        async def run_all() -> None:
            await asyncio.gather(
                *[
                    run_one(eid, job, mc, path, model, semaphores[model])
                    for model, work_items in work_items_dict.items()
                    for eid, job, mc, path in work_items
                ],
                return_exceptions=True,
            )
            logger.info(
                "Done: %d written, %d skipped, %d errors",
                EVAL_STATS["completed"],
                EVAL_STATS["skipped"],
                EVAL_STATS["errors"],
            )
            update_display(layout)
            live.update(layout)

        await asyncio.gather(run_all(), refresh_loop())
